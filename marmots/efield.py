"""
This class provides access to the parametrized electric field datafiles.
"""
import os.path as path
from typing import Any, Tuple

import attr
import numpy as np
from interpolation.splines import CGrid, eval_linear, extrap_options
from numba import njit
from scipy.interpolate import interpn

import marmots.geometry as geometry
from marmots import data_directory
from marmots.constants import Re

import os, sys



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
            The zenith angle of the shower (in degrees).
        decay: np.ndarray
            The decay altitude of each tau (in km).
        freqs: np.ndarray
            The frequency band (MHz).
        shower_energy: np.ndarray
            The energy of the shower.
        gain: float
            The peak gain [dBi].
        antennas: int
            The number of antennas.

        Returns
        -------
        Voltage: np.ndarray
            Returns the voltage (V) evaluated at each view angle.

        """

        voltage = np.zeros(view.size)

        too_far = decay_length > dbeacon
        
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

        sim_sinVB = interpn(
            (self.zenith_list[alt_idx], self.decay_list[alt_idx]),
            self.sim_sinVB[alt_idx],
            (exit_zenith, decay_altitude),
            bounds_error=False,
            fill_value=None,
        )

        sim_sinVB[sim_sinVB < 0] = 0
        
        mag, sinVB = geomag(self.bfield_grid, self.bfield, beacon, decay_zenith, decay_azimuth)

        efields = efield_interp(self.efield_grid[alt_idx], self.values[alt_idx], freqs, decay_altitude, exit_zenith, view)

        # calculate the voltage for each event
        voltage[~cut] = detector.voltage_from_field(
            efields,
            freqs,
            theta,
            (phi+360) % 360,
        )

        # account for ZHAIReS sims only extending to 3.16 deg in view angle
        #view_factor = np.ones(view.size)

        #view_factor[view > 3.16] = np.exp(
                    #-(view[view > 3.16])**2 / (2 * 3.16)**2
                #)

        #voltage[~cut] *= view_factor

        # distance correction (ZHAireS distance over Poinsseta distance)
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
            V = np.array([np.sin(np.deg2rad(zenith_decay)), np.zeros(zenith_decay.shape), np.cos(np.deg2rad(zenith_decay))]).T
            sinVB = geometry.norm(np.cross(V, B))

            sinVB = sinVB.reshape((self.zenith_list[i].size, self.decay_list[i].size, self.view_list[i].size))
                
            self.sim_sinVB.append(sinVB[:,:,0]) # sin(VxB) is independent of the view angle


    def load_file(self) -> None:
        """
        Load the parameterization file and store it into the class.
        """
        # load the data files

        self.altitudes = [0.5, 1.0, 2.0, 3.0, 4.0]

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
            interp_file = np.load(self.param_dir + f"/efield_lookup_{str(altitude)}km_v2.npz", allow_pickle=True)
        
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

    # loop over the array
    for i in np.arange(freqs.shape[-1]):

        # and perform the interpolation
        out[i, :] = eval_linear(
            grid,
            values,
            np.column_stack(
                (np.repeat(freqs[i], zenith.shape[-1]), decay, zenith, view)
            ),
            extrap_options.LINEAR
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
        np.array([lat, lon]),
        extrap_options.LINEAR
    )

    # and we are done
    return out


def geomag(
    grid, values, station: np.ndarray, zenith: np.ndarray, azimuth: np.ndarray
) -> np.ndarray:
    
    B = interp_bfield(grid, values, station[0], station[1])
    
    mag = np.linalg.norm(B)
    
    V = np.empty((zenith.size,3))
    V[:,0] = np.sin(np.deg2rad(zenith))*np.cos(np.deg2rad(azimuth))
    V[:,1] = np.sin(np.deg2rad(zenith))*np.sin(np.deg2rad(azimuth))
    V[:,2] = np.cos(np.deg2rad(zenith))
    
    sinVB = geometry.norm(np.cross( V, B/mag))
    
    return mag, sinVB

