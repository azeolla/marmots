"""
This class provides access to the parametrized electric field datafiles.
"""
import os.path as path
from typing import Any, Tuple

import attr
import numpy as np
from interpolation.splines import CGrid, eval_linear, extrap_options
from numba import njit
from scipy.interpolate import interp1d

import marmots.geometry as geometry
from marmots import data_directory
from marmots.constants import Re

import os, sys

import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter('ignore', NumbaDeprecationWarning)
warnings.simplefilter('ignore', NumbaPendingDeprecationWarning)


@attr.s
class EFieldParam():
    """
    Load and sample the included BEACON E-field parameterization files.
    """

    # the directory where we store parameterizations
    param_dir = path.join(data_directory, "beacon")

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
        beacon: np.ndarray,
        dbeacon: np.ndarray,
        freqs: np.ndarray,
        shower_energy: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        FoV: float,
        detector,
    ) -> np.ndarray:
        """
        Evaluate the peak electric field from this parameterization at
        a given off-axis angle (in degrees) given the zenith angle
        (in degrees) and decay altitude (in km) of the tau, and the frequency (in MHz).
        From this efield, calculate the voltage.

        Parameters
        ----------
        view: np.ndarray
            An array of view angles w.r.t the shower axis (degrees)
        exit_zenith: np.ndarray
            The zenith angle of the tau at the exit point (in degrees).
        decay_altitude: np.ndarray
            The decay altitude of each tau (in km).
        decay_length: np.ndarray
            The distance from exit point to decay for each tau (in km).
        decay_zenith: np.ndarray
            The zenith angle of the EAS at the decay point (in degrees).
        decay_azimuth: np.ndarray
            The azimuth angle of the EAS at the decay point (in degrees).
        distance_to_decay: np.ndarray
            Distance from the station to each decay point (in km).
        detector_altitude: float
            Station altitude (in km).
        beacon: np.ndarray
            Geocentric coordinates of the station.
        dbeacon: np.ndarray,
            Distance from the station to each exit point.
        freqs: np.ndarray
            Frequencies at which to evaluate the voltage (MHz).
        shower_energy: np.ndarray
            The energy of the shower.
        theta: np.ndarray
            Observation zenith angle from the perspective of the station.
        phi: np.ndarray
            Observation azimuth angle from the perspective of the station.
        FoV: float,
            Station field-of-view.
        detector: class
            The Detector class (from antenna.py).

        Returns
        -------
        Voltage: np.ndarray
            Returns the peak voltage (V) associated with each event.

        """

        voltage = np.zeros(view.size)

        # these events decay behind the station
        too_far = decay_length > dbeacon

        # these decays occur outside the field-of-view
        outside_fov = ((phi < -FoV/2) | (phi > FoV/2))
        
        cut = np.logical_or(too_far, outside_fov)

        view = view[~cut]
        exit_zenith = exit_zenith[~cut]
        decay_altitude = decay_altitude[~cut]
        decay_length = decay_length[~cut]
        decay_zenith = decay_zenith[~cut]
        decay_azimuth = decay_azimuth[~cut]
        shower_energy = shower_energy[~cut]
        theta = theta[~cut]
        phi = phi[~cut]
        distance_decay_km = distance_to_decay[~cut]

        # Find the look-up table with the closest altitude (rounding up)
        alt_idx = np.where(self.altitudes >= detector_altitude)[0][0]
         
        # interpolate to find the distance from decay to detector in ZHAireS
        sim_distance_decay_km = distance_interp(
            self.dist_grid[alt_idx],
            self.Dsim[alt_idx],
            decay_altitude, 
            exit_zenith, 
            view,
        )
        
        sim_distance_decay_km[sim_distance_decay_km < 0] = 0

        # interpolate sin(VxB) in ZHAireS
        sinVB = interp1d(
            self.zenith_list[alt_idx],
            self.sim_sinVB[alt_idx],
            bounds_error=False,
            fill_value="extrapolate",
        )
        sim_sinVB = sinVB(exit_zenith)
        sim_sinVB[sim_sinVB < 0] = 0 # shouldn't ever be negative

        sim_sinVB = sinVB(exit_zenith)

        sim_sinVB[sim_sinVB < 0] = 0
        
        # calculate actual sin(VxB)
        mag, sinVB = geomag(self.bfield_grid, self.bfield, beacon, decay_zenith, decay_azimuth)

        # electric field look-up table interpolation
        efields = efield_interp(self.efield_grid[alt_idx], self.values[alt_idx], freqs, decay_altitude, exit_zenith, view)

        # calculate the voltage for each event
        voltage[~cut] = detector.voltage_from_field(
            efields,
            freqs,
            theta,
            (phi+360) % 360,
        )

        # account for ZHAIReS sims only extending to 3.16 deg in view angle
        view_factor = np.ones(view.size)

        view_factor[view > 3.16] = np.exp(
                    -(view[view > 3.16])**2 / (2 * 3.16)**2
                )

        voltage[~cut] *= view_factor

        # distance correction (ZHAireS distance over MARMOTS distance)
        voltage[~cut] *= (sim_distance_decay_km / distance_decay_km)

        # energy scaling
        voltage[~cut] *= (shower_energy / self.sim_energy)
        
        # correct for changing magnetic field and azimuth
        voltage[~cut] *= (mag/self.sim_Bmag * sinVB/sim_sinVB)

        # replace NaNs with zeros
        voltage[np.isnan(voltage)] = 0

        return voltage

    def __attrs_post_init__(self) -> None:
        """
        Called at the end of __init__. Currently just loads the data file.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.load_file()

        # we now construct the distance LUT for the electric field scaling

        # we now construct the distance LUT for the electric field scaling

        self.sim_Bmag = 56000
        sim_incl = 63.5
        self.sim_sinVB = []

        for i in range(len(self.altitudes)):

            B = np.array([np.cos(np.deg2rad(sim_incl)), 0, -np.sin(np.deg2rad(sim_incl))])
            V = np.array([np.sin(np.deg2rad(self.zenith_list[i])), np.zeros(self.zenith_list[i].shape), np.cos(np.deg2rad(self.zenith_list[i]))]).T
            sinVB = geometry.norm(np.cross(V, B))

            self.sim_sinVB.append(sinVB)


    def load_file(self) -> None:
        """
        Load the parameterization file and store it into the class.
        """
        # load the data files

        self.altitudes = [1.0, 2.0, 3.0, 4.0]

        self.values = []
        self.decay_list = []
        self.zenith_list = []
        self.view_list = []
        self.efield_grid = []
        self.dist_grid = []
        self.Dsim = []

        self.sim_icethick = 0.0
        self.sim_energy = 1e17
        
        geomag_file = np.load(self.param_dir + f"/geomagnetic.npz", allow_pickle=True)
        self.bfield_grid = CGrid(geomag_file["lat"], geomag_file["lon"])
        self.bfield = geomag_file["bfield"]

        for altitude in self.altitudes:
            interp_file = np.load(self.param_dir + f"/efield_lookup_{str(altitude)}km.npz", allow_pickle=True)
        
            grid = interp_file["grid"]
            self.values.append(interp_file["efield"])
            self.Dsim.append(interp_file["distance"])

            freqs = grid[0]
            self.decay_list.append(grid[1])
            self.zenith_list.append(grid[2])
            self.view_list.append(grid[3])

            self.efield_grid.append(CGrid(freqs, grid[1], grid[2], grid[3]))
            self.dist_grid.append(CGrid(grid[1], grid[2], grid[3]))


@njit
def distance_interp(
    grid: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    values: np.ndarray,
    decay: np.ndarray,
    zenith: np.ndarray,
    view: np.ndarray,
) -> np.ndarray:
    """
    Perform a multi-dimensional linear interpolation using Numba.

    Parameters
    ----------
    grid: CGrid
        The rectangular grid for the interpolation.
    values: np.ndarray
        The 4D array of values at the grid locations.
    decay: np.ndarray
        The decay altitudes to interpolate at (km).
    zenith: np.ndarray
        The zenith angles to interpolate at (degrees).
    view: np.ndarray
        The view to interpolate at (degrees).

    Returns
    -------
    distance: np.ndarray
       The distance from decay to detector given the exit zenith angle, decay altitude, and view angle.
    """
    # Perform the interpolation
    out = eval_linear(
        grid,
        values,
        np.column_stack(
            (decay, zenith, view)
        ),
        extrap_options.LINEAR
    )

    # and we are done
    return out


@njit
def efield_interp(
    grid: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    values: np.ndarray,
    freqs: np.ndarray,
    decay: np.ndarray,
    zenith: np.ndarray,
    view: np.ndarray,
) -> np.ndarray:
    """
    Perform a multi-dimensional linear interpolation using Numba.

    Parameters
    ----------
    grid: CGrid
        The rectangular grid for the interpolation.
    values: np.ndarray
        The 4D array of values at the grid locations.
    freqs: np.ndarray
        The frequencies to interpolate at (MHz).
    decay: np.ndarray
        The decay altitudes to interpolate at (km).
    zenith: np.ndarray
        The zenith angles to interpolate at (degrees).
    view: np.ndarray
        The view to interpolate at (degrees).

    Returns
    -------
    Efield: np.ndarray
       The electric field interpolated at each (f, d, z, v).
    """
    # allocate the output array
    out = np.empty((freqs.shape[-1], zenith.shape[-1]), dtype=np.float64)

    clipped = np.copy(view)
    clipped[view > 3.16] = 3.16 # strange extrapolation beyond 3.16 degrees

    # loop over the array
    for i in np.arange(freqs.shape[-1]):

        # and perform the interpolation
        out[i, :] = eval_linear(
            grid,
            values,
            np.column_stack(
                (np.repeat(freqs[i], zenith.shape[-1]), decay, zenith, clipped) 
            )
        )

    # and we are done
    return out
    

@njit
def interp_bfield(
    grid: Tuple[np.ndarray, np.ndarray],
    values: np.ndarray,
    lat: float,
    lon: float,
) -> np.ndarray:
    """
    Perform a multi-dimensional linear interpolation using Numba.

    Parameters
    ----------
    grid: CGrid
        The rectangular grid for the interpolation.
    values: np.ndarray
        The 2D array of values at the grid locations.
    lat: np.ndarray
        The latitude to interpolate at (deg).
    lon: np.ndarray
        The longitude to interpolate at (deg).

    Returns
    -------
    out: np.ndarray
       The geomagnetic field at the given latitude and longitude in ENU coordinates.
    """
    # Perform the interpolation
    out = eval_linear(
        grid,
        values,
        np.array([lat, lon]),
    )

    # and we are done
    return out


def geomag(
    grid, values, station: np.ndarray, zenith: np.ndarray, azimuth: np.ndarray
) -> np.ndarray:
    """
    Given the station location, and event geometry, returns the magnetic field strength and sin(VxB).
    """
    
    B = interp_bfield(grid, values, station[0], station[1])
    
    mag = np.linalg.norm(B)
    
    V = np.empty((zenith.size,3))
    V[:,0] = np.sin(np.deg2rad(zenith))*np.cos(np.deg2rad(azimuth))
    V[:,1] = np.sin(np.deg2rad(zenith))*np.sin(np.deg2rad(azimuth))
    V[:,2] = np.cos(np.deg2rad(zenith))
    
    sinVB = geometry.norm(np.cross( V, B/mag))
    
    return mag, sinVB

