"""
marmots/tau_propagator.py

Propagate tau leptons from exit point to decay point through topography.
For tracks that cross terrain (tau_path_clear = False):
    1. Find alternating air/rock segments
    2. Compute survival probability through each segment
    3. Re-sample decay point in the last air segment with E_final
    4. Return Psurv_tot, E_final, new decay point

Only called for tracks where tau_path_clear() returns False.
"""

import numpy as np
import marmots.mesh as mesh
from scipy.interpolate import RegularGridInterpolator
from marmots import data_directory


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RHO_ROCK = 2.65        # g/cm³  — consistent with Psurv_table generation
CTAU_KM  = 87.03e-6    # km  (tau decay length at E = m_tau)
M_TAU    = 1.77686     # GeV


# ---------------------------------------------------------------------------
# Energy loss parameters (ALLM and ASW models)
# ---------------------------------------------------------------------------

Eloss_pars = np.array([
    [ 2.05820774222e-07,  4.93367455295e-09, 0.2277817378870],  # ALLM
    [-4.77043758142e-08,  1.90315208270e-07, 0.0469916563971]   # ASW
])


# ---------------------------------------------------------------------------
# Survival LUT
# ---------------------------------------------------------------------------

class TauSurvLUT:
    """
    Look-Up Table for tau survival probability in rock.
    Interpolates Psurv(E_GeV, d_km) from the NES_QUICK precomputed table.

    Table format (3 columns):
        log10(E / GeV)   log10(d / km)   Psurv

    Usage
    -----
    lut   = TauSurvLUT()
    psurv = lut(E_GeV, d_km)   # scalars or shape-(N,) arrays
    """

    def __init__(self, filename: str = "/taupropagation/Psurv_table.txt"):

        data = np.loadtxt(data_directory + filename, comments='#')

        log_E_vals = np.unique(data[:, 0])   # shape (11,)
        log_d_vals = np.unique(data[:, 1])   # shape (100000,)

        nE = log_E_vals.size
        nd = log_d_vals.size

        Psurv_grid = data[:, 2].reshape(nE, nd)

        self._interp = RegularGridInterpolator(
            (log_E_vals, log_d_vals),
            Psurv_grid,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        self._log_E_min = log_E_vals[0]
        self._log_E_max = log_E_vals[-1]
        self._log_d_min = log_d_vals[0]
        self._log_d_max = log_d_vals[-1]

    def __call__(self, E_GeV, d_km) -> np.ndarray:
        """
        Parameters
        ----------
        E_GeV : float or np.ndarray  — tau energy [GeV], linear scale
        d_km  : float or np.ndarray  — rock distance [km]

        Returns
        -------
        Psurv : np.ndarray clipped to [0, 1]
        """
        E_GeV = np.atleast_1d(np.asarray(E_GeV, dtype=float))
        d_km  = np.atleast_1d(np.asarray(d_km,  dtype=float))

        d_km  = np.where(d_km  > 0, d_km,  1e-10)
        E_GeV = np.where(E_GeV > 0, E_GeV, 1e-10)

        log_E = np.clip(np.log10(E_GeV), self._log_E_min, self._log_E_max)
        log_d = np.clip(np.log10(d_km),  self._log_d_min, self._log_d_max)

        pts = np.column_stack([log_E, log_d])
        return np.clip(self._interp(pts), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Survival probability in air
# ---------------------------------------------------------------------------

def psurv_air(E_GeV: float, L_km: float) -> float:
    """
    Probability that a tau of energy E_GeV survives (does not decay)
    while travelling L_km in air.

    Psurv_air = exp(-L / L_dec(E))
    with L_dec(E) = E/m_tau * ctau  [km]
    """
    L_dec = (E_GeV / M_TAU) * CTAU_KM
    return float(np.exp(-L_km / L_dec))


# ---------------------------------------------------------------------------
# Energy loss formula
# ---------------------------------------------------------------------------

def tau_energy_loss(x: float,
                    E0: float,
                    a0: float,
                    a1: float,
                    a2: float) -> float:
    """
    Tau energy [GeV] after traversing column depth x [g/cm²] in rock.
    dE/dX = -b(E)*E,  b(E) = a0 + a1*(E/GeV)^a2   (FROM NES_QUICK)
    """
    return float(
        E0 * np.exp(-a0 * x) /
        ((1.0 + a1 * (E0 ** a2) * (1.0 - np.exp(-a2 * a0 * x)) / a0) ** (1.0 / a2))
    )


# ---------------------------------------------------------------------------
# Segment finder
# ---------------------------------------------------------------------------

def get_air_rock_segments(exitpoint: np.ndarray,
                          decaypoint: np.ndarray,
                          TotalArea,
                          BVH) -> list:
    """
    Find alternating air/rock segments between exit and decay points.
    The tau just exited the Earth so the first segment is always air.

    Parameters
    ----------
    exitpoint  : shape (3,) geocentric [km]
    decaypoint : shape (3,) geocentric [km]

    Returns
    -------
    list of dicts with keys:
        'type'   : 'air' or 'rock'
        'start'  : np.ndarray (3,) geocentric [km]
        'end'    : np.ndarray (3,) geocentric [km]
        'length' : float [km]
    """
    direction  = decaypoint - exitpoint
    total_dist = np.linalg.norm(direction)
    axis       = direction / total_dist

    origin = TotalArea.center

    # correct transformation of direction vector into local coords
    local_exit = mesh.geocentric2local(exitpoint[None, :], origin)[0]
    local_axis = mesh.geocentric2local((origin + axis)[None, :], origin)[0]
    local_axis /= np.linalg.norm(local_axis)

    # forward ray intersections with the mesh
    local_ints = np.array(mesh.intersect_bvh(BVH, local_exit, local_axis))

    # pure air — no intersections
    if local_ints.size == 0:
        return [{"type": "air", "start": exitpoint,
                 "end": decaypoint, "length": total_dist}]

    ints      = mesh.local2geocentric(local_ints, origin)
    distances = np.linalg.norm(ints - exitpoint, axis=1)

    # keep only intersections strictly between exit and decay
    mask = (distances > 1e-3) & (distances < total_dist - 1e-3)

    if not np.any(mask):
        return [{"type": "air", "start": exitpoint,
                 "end": decaypoint, "length": total_dist}]

    ints      = ints[mask]
    distances = distances[mask]
    order     = np.argsort(distances)
    ints      = ints[order]

    segments   = []
    prev_point = exitpoint
    in_air     = True   # tau exits Earth → first segment always air

    for p in ints:
        length = np.linalg.norm(p - prev_point)
        if length > 1e-6:
            segments.append({
                "type":   "air" if in_air else "rock",
                "start":  prev_point,
                "end":    p,
                "length": length
            })
        in_air     = not in_air
        prev_point = p

    length = np.linalg.norm(decaypoint - prev_point)
    if length > 1e-6:
        segments.append({
            "type":   "air" if in_air else "rock",
            "start":  prev_point,
            "end":    decaypoint,
            "length": length
        })

    return segments


# ---------------------------------------------------------------------------
# Single track propagator
# ---------------------------------------------------------------------------

def propagate_single_track(segments: list,
                           E0: float,
                           psurvival_lut: TauSurvLUT,
                           a0: float, a1: float, a2: float) -> tuple:
    """
    Propagate one tau through alternating air/rock segments.

    Computes:
        Psurv_tot = Psurv_air1(E0) * Psurv_rock1(E0) *
                    Psurv_air2(E1) * Psurv_rock2(E1) * ...
        E_final   = E0 after all rock energy losses
        last_air_start = start point of the last air segment [geocentric km]

    Parameters
    ----------
    segments      : output of get_air_rock_segments
    E0            : tau energy at exit point [GeV]
    psurvival_lut : TauSurvLUT instance
    a0, a1, a2    : energy loss parameters

    Returns
    -------
    Psurv_tot      : float ∈ [0, 1]
    E_final        : float [GeV]
    last_air_start : np.ndarray shape (3,) geocentric [km]
    """
    E         = E0
    P_total   = 1.0

    last_air_start = segments[0]["start"]   # exit point by default

    for seg in segments:
        L        = seg["length"]
        material = seg["type"]

        if material == "air":
            P_total       *= psurv_air(E, L)
            last_air_start = seg["start"]

        else:  # rock
            grammage = L * 1e5 * RHO_ROCK          # g/cm²
            P_total *= float(psurvival_lut(E, L))
            E        = tau_energy_loss(grammage, E, a0, a1, a2)

        # early exit if tau is essentially absorbed
        if P_total < 1e-10:
            return 0.0, E, last_air_start

    return P_total, E, last_air_start


# ---------------------------------------------------------------------------
# Vectorised interface for effective_area.py
# ---------------------------------------------------------------------------

def propagate_tracks(exit_points:   np.ndarray,
                     decay_points:  np.ndarray,
                     Etau:          np.ndarray,
                     TotalArea,
                     BVH,
                     psurvival_lut: TauSurvLUT,
                     taudecay,
                     axis:          np.ndarray,
                     model:         int = 0) -> tuple:
    """
    For every track where tau_path_clear = False, propagate through terrain
    and return corrected quantities.

    Parameters
    ----------
    exit_points   : shape (N, 3) geocentric [km]
    decay_points  : shape (N, 3) geocentric [km]  — original sampled
    Etau          : shape (N,)   tau energies [GeV]
    TotalArea     : NamedTuple with .center
    BVH           : BVHNode
    psurvival_lut : TauSurvLUT instance
    taudecay      : DecayParticle instance (from pythia.py)
    axis          : shape (3,)  particle axis unit vector
    model         : 0=ALLM, 1=ASW

    Returns
    -------
    Psurv_tot    : shape (N,)  weight to multiply into Pexit
    E_final      : shape (N,)  tau energy [GeV] entering last air segment
    decay_points : shape (N, 3) corrected decay points [geocentric km]
    """
    a0, a1, a2 = Eloss_pars[model]

    N          = exit_points.shape[0]
    Psurv_tot  = np.ones(N)
    E_final    = Etau.copy()
    decay_pts  = decay_points.copy()

    for k in range(N):
        segs = get_air_rock_segments(
            exit_points[k], decay_points[k], TotalArea, BVH
        )

        # skip pure-air tracks — should not reach here if called after
        # tau_path_clear, but kept as a safety check
        if all(s["type"] == "air" for s in segs):
            continue

        P, E_f, last_air_start = propagate_single_track(
            segs, Etau[k], psurvival_lut, a0, a1, a2
        )

        Psurv_tot[k] = P
        E_final[k]   = E_f

        # re-sample decay length in the last air segment with E_final
        new_decay_length      = float(taudecay.sample_range(
                                    np.array([E_f]))[0])
        decay_pts[k]          = last_air_start + axis * new_decay_length

    return Psurv_tot, E_final, decay_pts