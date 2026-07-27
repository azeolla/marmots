"""
This class provides access to the parametrized electric field datafiles.

Drop-in replacement for the original EFieldParam. The new LUT is indexed by
(freq, hdecay, theta_decay, distance_to_decay, view) and stores either:
  - RxEfield_TauLUT_MARMOTS_geomagnetic_askaryan.npz : d_tau * E_peak  [m·V/m]
  - Efield_TauLUT_MARMOTS_geomagnetic_askaryan.npz   : E_peak          [V/m]

Both files contain separate arrays for geomagnetic and Askaryan components.
The two are combined vectorially as:
    E^2 = E_G^2 + E_A^2 + 2*E_G*E_A*cos(eta)
where eta is the angle between the geomagnetic polarisation direction (V×B)
and the unit vector from the decay point to the station (r_hat_{tau->Ant}).
"""
import os.path as path
from typing import Tuple

import attr
import numpy as np
from interpolation.splines import CGrid, eval_linear, extrap_options
from numba import njit

import marmots.geometry as geometry
from marmots import data_directory
from marmots.constants import Re

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


@attr.s
class EFieldParam():
    """
    Load and sample the BEACON E-field parameterization.

    LUT axes: (freq [MHz], hdecay [km], theta_decay [deg], distance [km], view [deg])

    ZHAireS simulation parameters set in load_file():
        sim_energy    = 0.98e17 eV
        sim_Bmag      = 50000 nT  (50 muT)
        alpha         = 90 deg (angle between shower axis and B field in sims)
        sim_Bsinalpha = sim_Bmag * sin(90 deg) = 50000 nT
    """

    param_dir = path.join(data_directory, "e_field_library")

    def __call__(
        self,
        view: np.ndarray,
        exit_zenith: np.ndarray,
        decay_altitude: np.ndarray,
        decay_length: np.ndarray,
        decay_zenith: np.ndarray,
        decay_azimuth: np.ndarray,
        distance_to_decay: np.ndarray,
        detector_altitude: float,
        observatory: np.ndarray,
        dobservatory: np.ndarray,
        freqs: np.ndarray,
        shower_energy: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        FoV: float,
        detector,
        decay_point: np.ndarray,
        observatory_geocentric: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the peak voltage for each event.

        Parameters
        ----------
        view: np.ndarray
            Off-axis angle w.r.t. shower axis (degrees).
        exit_zenith: np.ndarray
            Zenith angle of the tau at exit point (degrees).
        decay_altitude: np.ndarray
            Decay altitude of each tau (km).
        decay_length: np.ndarray
            Distance from exit point to decay (km).
        decay_zenith: np.ndarray
            Zenith angle of the EAS at decay point (degrees).
        decay_azimuth: np.ndarray
            Azimuth angle of the EAS at decay point (degrees).
        distance_to_decay: np.ndarray
            Distance from station to each decay point (km).
        detector_altitude: float
            Station altitude (km).
        observatory: np.ndarray
            Geodetic coordinates of the station [lat, lon, alt].
        dobservatory: np.ndarray
            Distance from station to each exit point (km).
        freqs: np.ndarray
            Frequencies to evaluate (MHz).
        shower_energy: np.ndarray
            Shower energy (eV).
        theta: np.ndarray
            Observation zenith angle from station (degrees).
        phi: np.ndarray
            Observation azimuth angle from station (degrees).
        FoV: float
            Station field-of-view (degrees).
        detector: Detector
            Detector class from antenna.py.
        decay_point: np.ndarray
            Geocentric coordinates of decay points [x,y,z] in Re,
            shape (n_events, 3).
        observatory_geocentric: np.ndarray
            Geocentric coordinates of the station [x,y,z] in Re.

        Returns
        -------
        voltage: np.ndarray
            Peak voltage (V) for each event, shape (n_events,).
        """
        voltage = np.zeros(view.size)

        # ------------------------------------------------------------------ #
        # 1. Cuts: events that decay behind the station or outside the FoV
        # ------------------------------------------------------------------ #
        too_far     = decay_length > dobservatory
        outside_fov = (phi < -FoV / 2) | (phi > FoV / 2)
        cut = too_far | outside_fov

        view_c      = view[~cut]
        exit_zen_c  = exit_zenith[~cut]
        decay_alt_c = decay_altitude[~cut]
        decay_zen_c = decay_zenith[~cut]
        decay_az_c  = decay_azimuth[~cut]
        dist_c      = distance_to_decay[~cut]   # km
        energy_c    = shower_energy[~cut]
        theta_c     = theta[~cut]
        phi_c       = phi[~cut]
        decay_point_c = decay_point[~cut]       # geocentric, shape (n_ev, 3)

        n_ev = view_c.size
        if n_ev == 0:
            return voltage

        # ------------------------------------------------------------------ #
        # 2. Interpolate geomagnetic and Askaryan components from 5D LUT
        #    Axes: (freq, hdecay, theta_decay, distance, view)
        #    Output shape: (n_freqs, n_events)
        # ------------------------------------------------------------------ #
        E_geo = efield_interp(
            self.lut_cgrid, self.values_geo,
            freqs, decay_alt_c, decay_zen_c, dist_c, view_c,
        )
        E_ask = efield_interp(
            self.lut_cgrid, self.values_ask,
            freqs, decay_alt_c, decay_zen_c, dist_c, view_c,
        )

        # ------------------------------------------------------------------ #
        # 3. If LUT stores R*E, divide by real distance to recover E [V/m]
        #    dist_c is in km; LUT was built with distance in m → divide by m
        # ------------------------------------------------------------------ #
        if self.is_RxE:
            dist_m = dist_c * 1000.0                         # km → m
            E_geo  = E_geo / dist_m[np.newaxis, :]
            E_ask  = E_ask / dist_m[np.newaxis, :]

        # ------------------------------------------------------------------ #
        # 4. Geomagnetic correction: scale by (B*sinVB)_real / (B*sinVB)_sim
        #    Only the geomagnetic component carries this dependence.
        # ------------------------------------------------------------------ #
        mag, sinVB = geomag(
            self.bfield_grid, self.bfield, observatory, decay_zen_c, decay_az_c
        )
        # mag [nT], sinVB shape (n_events,)
        real_Bsinalpha = mag * sinVB                          # (n_events,)
        geo_scale      = real_Bsinalpha / self.sim_Bsinalpha  # (n_events,)

        E_geo = E_geo * geo_scale[np.newaxis, :]              # (n_freqs, n_events)

        # ------------------------------------------------------------------ #
        # 5. Vectorial combination of the two polarization components
        #    E^2 = E_G^2 + E_A^2 + 2*E_G*E_A*cos(eta)
        #    eta = angle between r_hat_{tau->Ant} and V×B (geo polarisation)
        # ------------------------------------------------------------------ #
        cos_eta = compute_cos_eta(
            decay_zen_c, decay_az_c,
            self.bfield_grid, self.bfield,
            observatory,            # geodetic [lat, lon, alt] for B interpolation
            observatory_geocentric, # geocentric [x,y,z] for r_hat
            decay_point_c,          # geocentric [x,y,z], shape (n_ev, 3)
        )                                                      # (n_events,)

        E_combined = np.sqrt(
            np.clip(
                E_geo**2 + E_ask**2
                + 2.0 * E_geo * E_ask * cos_eta[np.newaxis, :],
                0.0, None,
            )
        )                                                      # (n_freqs, n_events)

        # ------------------------------------------------------------------ #
        # 6. Compute voltage via detector response  (same call as original)
        # ------------------------------------------------------------------ #
        volt = detector.voltage_from_field(
            E_combined,
            freqs,
            theta_c,
            (phi_c + 360) % 360,
        )                                                      # (n_events,)

        # ------------------------------------------------------------------ #
        # 7. View-angle suppression beyond LUT range  (same as original)
        # ------------------------------------------------------------------ #
        psi_max = float(self.lut_axes[4][-1])   # 3.0 deg
        view_factor = np.ones(n_ev)
        beyond = view_c > psi_max
        view_factor[beyond] = np.exp(
            -(view_c[beyond])**2 / (2.0 * psi_max)**2
        )
        volt *= view_factor

        # ------------------------------------------------------------------ #
        # 8. Energy scaling  (same as original)
        # ------------------------------------------------------------------ #
        volt *= energy_c / self.sim_energy

        # ------------------------------------------------------------------ #
        # 9. Store results, replace NaNs with zeros  (same as original)
        # ------------------------------------------------------------------ #
        voltage[~cut] = volt
        voltage[np.isnan(voltage)] = 0.0

        return voltage

    def __attrs_post_init__(self) -> None:
        """Called at end of __init__. Loads data files."""
        self.load_file()

    def load_file(self) -> None:
        """
        Load the LUT and geomagnetic field map, and store ZHAireS sim parameters.

        ZHAireS simulation parameters (not stored in the npz, set here):
            sim_energy    [eV]  : reference shower energy
            sim_Bmag      [nT]  : |B| of the simulated field (50 muT)
            sim_incl      [deg] : magnetic field inclination used in sims
            sim_Bsinalpha [nT]  : B * sin(alpha), the amplitude factor
                                  for geomagnetic scaling. alpha=90 deg in
                                  the simulations, so sin(alpha)=1.
        """
        # --- geomagnetic field lookup table ---
        geomag_file = np.load(
            path.join(self.param_dir, "geomagnetic.npz"), allow_pickle=True
        )
        self.bfield_grid = CGrid(geomag_file["lat"], geomag_file["lon"])
        self.bfield      = geomag_file["bfield"]

        # --- ZHAireS simulation parameters ---
        self.sim_energy    = 0.98e17   # eV  — reference shower energy
        self.sim_Bmag      = 50000.0   # nT  — |B| in ZHAireS sims (50 muT)
        self.sim_incl      = -45.0     # deg — magnetic inclination in sims
        # alpha = 90 deg in sims (shower perpendicular to B) → sin(alpha) = 1
        self.sim_Bsinalpha = self.sim_Bmag * np.sin(np.deg2rad(90.0))  # = 50000 nT

        # --- LUT: prefer RxEfield (better numerical conditioning) ---
        rxe_path = path.join(
            self.param_dir, "RxEfield_TauLUT_MARMOTS_geomagnetic_askaryan.npz"
        )
        e_path = path.join(
            self.param_dir, "Efield_TauLUT_MARMOTS_geomagnetic_askaryan.npz"
        )

        if path.exists(rxe_path):
            lut = np.load(rxe_path, allow_pickle=True)
            self.is_RxE = True
        elif path.exists(e_path):
            lut = np.load(e_path, allow_pickle=True)
            self.is_RxE = False
        else:
            raise FileNotFoundError(
                f"No LUT file found in {self.param_dir}.\n"
                "Expected RxEfield_TauLUT_MARMOTS_geomagnetic_askaryan.npz "
                "or Efield_TauLUT_MARMOTS_geomagnetic_askaryan.npz"
            )

        # grid axes: [freq(MHz), hdecay(km), theta_decay(deg), distance(km), view(deg)]
        self.lut_axes   = lut["grid"]
        self.values_geo = lut["efield_geomagnetic"]  # (n_freq, n_h, n_t, n_d, n_v)
        self.values_ask = lut["efield_askaryan"]

        # CGrid for numba interpolation
        self.lut_cgrid = CGrid(
            self.lut_axes[0].astype(float),
            self.lut_axes[1].astype(float),
            self.lut_axes[2].astype(float),
            self.lut_axes[3].astype(float),
            self.lut_axes[4].astype(float),
        )


# ========================================================================== #
#  Numba-accelerated 5D linear interpolation
# ========================================================================== #

@njit
def efield_interp(
    grid,
    values: np.ndarray,
    freqs: np.ndarray,
    decay: np.ndarray,
    zenith: np.ndarray,
    distance: np.ndarray,
    view: np.ndarray,
) -> np.ndarray:
    """
    5D linear interpolation over (freq, hdecay, theta_decay, distance, view).

    Parameters
    ----------
    grid: CGrid
        Built from (freq, hdecay, theta_decay, distance, view).
    values: np.ndarray
        Shape (n_freq, n_h, n_theta, n_d, n_v).
    freqs: np.ndarray
        Frequencies [MHz], shape (n_freqs,).
    decay: np.ndarray
        Decay altitudes [km], shape (n_events,).
    zenith: np.ndarray
        Zenith angles [deg], shape (n_events,).
    distance: np.ndarray
        Distances to decay [km], shape (n_events,).
    view: np.ndarray
        Off-axis angles [deg], shape (n_events,).

    Returns
    -------
    out: np.ndarray
        Shape (n_freqs, n_events).
    """
    n_freqs  = freqs.shape[-1]
    n_events = zenith.shape[-1]
    out = np.empty((n_freqs, n_events), dtype=np.float64)

    # clip view to LUT range
    psi = np.copy(view)
    psi_max = 3.0
    for k in range(n_events):
        if psi[k] > psi_max:
            psi[k] = psi_max

    for i in range(n_freqs):
        pts = np.column_stack((
            np.repeat(freqs[i], n_events),
            decay,
            zenith,
            distance,
            psi,
        ))
        out[i, :] = eval_linear(grid, values, pts)

    return out


# ========================================================================== #
#  Geomagnetic field helpers
# ========================================================================== #

@njit
def interp_bfield(
    grid,
    values: np.ndarray,
    lat: float,
    lon: float,
) -> np.ndarray:
    """Interpolate B-field vector at (lat, lon) in ENU coordinates."""
    return eval_linear(grid, values, np.array([lat, lon]))


def geomag(
    grid,
    values: np.ndarray,
    station: np.ndarray,
    zenith: np.ndarray,
    azimuth: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Return |B| [nT] and sin(V×B) for each event.
    """
    B   = interp_bfield(grid, values, station[0], station[1])
    mag = np.linalg.norm(B)

    V = np.empty((zenith.size, 3))
    V[:, 0] = np.sin(np.deg2rad(zenith)) * np.cos(np.deg2rad(azimuth))
    V[:, 1] = np.sin(np.deg2rad(zenith)) * np.sin(np.deg2rad(azimuth))
    V[:, 2] = np.cos(np.deg2rad(zenith))

    sinVB = geometry.norm(np.cross(V, B / mag))

    return mag, sinVB


def compute_cos_eta(
    decay_zen: np.ndarray,
    decay_az: np.ndarray,
    bfield_grid,
    bfield: np.ndarray,
    observatory_geodetic: np.ndarray,
    observatory_geocentric: np.ndarray,
    decay_point: np.ndarray,
) -> np.ndarray:
    """
    cos(eta) = r_hat_{tau->Ant} · (V×B) / |V×B|

    eta is the angle between the unit vector from the decay point to the
    station and the geomagnetic polarisation direction V×B.

    Parameters
    ----------
    decay_zen             : np.ndarray — shower zenith at decay (deg), (n_events,)
    decay_az              : np.ndarray — shower azimuth at decay (deg), (n_events,)
    bfield_grid, bfield   : geomagnetic field LUT
    observatory_geodetic  : np.ndarray — [lat, lon, alt] for B interpolation
    observatory_geocentric: np.ndarray — [x, y, z] in Re for r_hat computation
    decay_point           : np.ndarray — [x, y, z] in Re, shape (n_events, 3)

    Returns
    -------
    cos_eta : np.ndarray, shape (n_events,)
    """
    # B field unit vector at the station
    B    = interp_bfield(bfield_grid, bfield,
                         observatory_geodetic[0], observatory_geodetic[1])
    Bhat = B / np.linalg.norm(B)

    n_ev    = decay_zen.size
    cos_eta = np.zeros(n_ev)

    for i in range(n_ev):

        # vector 
        r_vec = observatory_geocentric - decay_point[i]
        r_hat = r_vec / np.linalg.norm(r_vec)

        #  shower direction
        V = np.array([
            np.sin(np.deg2rad(decay_zen[i])) * np.cos(np.deg2rad(decay_az[i])),
            np.sin(np.deg2rad(decay_zen[i])) * np.sin(np.deg2rad(decay_az[i])),
            np.cos(np.deg2rad(decay_zen[i])),
        ])

        #
        VxB  = np.cross(V, Bhat)
        e1   = VxB / np.linalg.norm(VxB)          #geomag direction (V×B̂)
        Ve1  = np.cross(V, e1)
        e2   = Ve1 / np.linalg.norm(Ve1)           

        # shower plane proyection of r_hat 
        beta  = np.dot(r_hat, e1)
        delta = np.dot(r_hat, e2)

        # cos(eta) angle inside the shower plane 
        proj_norm = np.sqrt(beta**2 + delta**2)
        if proj_norm > 0:
            cos_eta[i] = beta / proj_norm
        else:
            cos_eta[i] = 0.0

    return cos_eta

