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
from igrf12 import igrf

import marmots.antenna as antenna
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
        beacon: dict,
        dbeacon: np.ndarray,
        freqs: np.ndarray,
        shower_energy: np.ndarray,
        antennas: int,
        theta: np.ndarray,
        phi: np.ndarray,
        FoV: float,
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
        # clip the arrays to the ZHArieS sim bounds
        #zenith = np.clip(exit_zenith, 55.0, 89.0)
        #view = np.clip(view, 0.04, 3.16)
        #decay = np.clip(decay_altitude, 0, self.sim_altitude - 0.5)

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

        alt_idx = np.abs(self.altitudes - detector_altitude).argmin()
         

        # interpolate to find the distance from decay to detector in ZHAireS
        sim_distance_decay_km = interpn(
            (self.zenith_list[alt_idx], self.decay_list[alt_idx], self.view_list[alt_idx]),
            self.Dsim[alt_idx],
            (exit_zenith, decay_altitude, view),
            bounds_error=False,
            fill_value=None,
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
        
        mag, sinVB = geomag(beacon, decay_zenith, decay_azimuth)

        efields = eval(self.grid[alt_idx], self.values[alt_idx], freqs, decay_altitude, exit_zenith, view).T

        # calculate the voltage at each frequency
        voltage[~cut] = np.sum(antenna.voltage_from_field(
            efields,
            freqs,
            antennas,
            theta,
            (phi+360) % 360,
        ), axis=1)

        # account for ZHAIReS sims only extending to 3.16 deg in view angle
        view_factor = np.ones(view.size)

        view_factor[view > 3.16] = np.exp(
                    -(view[view > 3.16])**2 / (2 * 3.16)**2
                )

        voltage[~cut] *= view_factor

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

        self.Dsim = []
        self.sim_Bmag = 56000
        sim_incl = 63.5
        self.sim_sinVB = []

        for i in range(len(self.altitudes)):

            # mesh the loaded decay altitudes and zenith angles
            Da, Za, Va = np.meshgrid(self.decay_list[i], self.zenith_list[i], self.view_list[i])

            # calculate the distance to the detector in each sim
            Dsim, zenith_decay = distance_decay_to_detector_LUT(
                Da.flatten(), Za.flatten(), Va.flatten(), self.altitudes[i], self.sim_icethick
            )

            # reshape the array to the appropriate size and save
            self.Dsim.append(Dsim.reshape((self.zenith_list[i].size, self.decay_list[i].size, self.view_list[i].size)))

            B = np.array([np.cos(np.deg2rad(sim_incl)), 0, -np.sin(np.deg2rad(sim_incl))])
            V = np.array([np.sin(np.deg2rad(zenith_decay)), np.zeros(zenith_decay.shape), np.cos(np.deg2rad(zenith_decay))]).T
            sinVB = np.linalg.norm(np.cross(V, B), axis=1)

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
        self.grid = []

        self.sim_icethick = 0.0
        self.sim_energy = 1e17

        for altitude in self.altitudes:
            interp_file = np.load(self.param_dir + f"/efield_lookup_{str(altitude)}km.npz", allow_pickle=True)
        
            grid = interp_file["grid"]
            self.values.append(interp_file["efield"])

            freqs = grid[0]
            self.decay_list.append(grid[1])
            self.zenith_list.append(grid[2])
            self.view_list.append(grid[3])

            self.grid.append(CGrid(freqs, grid[1], grid[2], grid[3]))

            


@njit
def eval(
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
    out = np.zeros((freqs.shape[-1], zenith.shape[-1]), dtype=np.float64)

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

def get_X0(
    zenith: np.ndarray, decay_altitude: np.ndarray, ice: float = 0.0
) -> np.ndarray:
    """
    Compute the location of X0 given the event parameters.
    See @swissel for details about this implementation.
    Parameters
    ----------
    ice: float
        The ice thickness in km.
    decay_altitude: np.ndarray
        The decay altitude in km.
    zenith: np.ndarray
        The zenith angle in degrees.
    """
    a = 1.0
    b = 2.0 * np.cos(np.deg2rad(zenith)) * (Re + ice)
    c = (Re + ice) ** 2 - (Re + ice + decay_altitude) ** 2
    X0 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return X0


def get_decay_zenith_angle(
    decay_altitude: np.ndarray, X0: np.ndarray, ice: float = 0.0
) -> np.ndarray:
    """
    Get the zenith angle at the decay point.
    See @swissel for details about this implementation.
    Parameters
    ----------
    decay_altitude: np.ndarray
        The decay altitude (in km).
    X0: np.ndarray
        The shower origin location.
    ice: float
        The thickness of the ice (km).
    """

    # get the quantities for the cosine rule
    A = X0
    B = Re + ice + decay_altitude
    C = Re + ice

    # construct cosz
    cosz = (A ** 2 + B ** 2 - C ** 2) / (2 * A * B)
    return np.rad2deg(np.arccos(cosz))


def get_distance_decay_to_detector(
    zenith_decay: np.ndarray,
    view: np.ndarray,
    decay_altitude: np.ndarray,
    detector_altitude: float = 37.0,
    ice: float = 0.0,
) -> np.ndarray:
    """
    Get the distance from the decay point to the detector.
    See @swissel for details on this implementation.
    Parameters
    ----------
    decay_altitude: np.ndarray
        The decay altitude (in km).
    detector_altitude:
        The altitude of the detector in (km).
    zenith_decay: np.ndarray
        The zenith angle at the decay (degrees)
    view: np.ndarray
        The view angle to the detector (degrees)
    ice: float
        The thickness of the ice (km).
    """
    a = 1
    b = 2 * np.cos(np.deg2rad(zenith_decay)) * (Re + ice + decay_altitude)
    c = (Re + ice + decay_altitude) ** 2 - (Re + detector_altitude) ** 2
    d = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    dist = d/np.cos(np.deg2rad(view))
    return dist


def distance_decay_to_detector_LUT(
    decay_altitude: np.ndarray,
    zenith: np.ndarray,
    view: np.ndarray,
    detector_altitude: float = 37.0,
    ice: float = 0.0,
) -> np.ndarray:
    """
    Return the distance from the decay to the detector (km).
    See @swissel for details of this implementation.
    Parameters
    ----------
    decay_altitude: np.ndarray
        The decay altitude (km).
    detector_altitude: float
        The altitude of the detector (km).
    zenith: np.ndarray
        The zenith angle (degrees).
    ice: float
        The ice thickness (km)
    Returns
    -------
    distance: np.ndarray
        The distance from the decay to the detector (km).
    """

    # we coerce floats to numpy arrays
    zenith = np.atleast_1d(zenith)
    decay_altitude = np.atleast_1d(decay_altitude)

    # construct the distance array
    distance = np.zeros_like(zenith)
    zenith_decay = np.zeros_like(zenith)

    # find the index of ground events
    ground = decay_altitude < 1e-10

    zenith_decay[ground] = zenith[ground]
    # finds the distance between the two points defined by the
    # decay altitude, detector altitude, and the zenith angle at the point
    distance[ground] = get_distance_decay_to_detector(zenith_decay[ground], view[ground], decay_altitude[ground], detector_altitude, ice)

    # get the distance to the decay
    X0 = get_X0(zenith[~ground], decay_altitude[~ground], ice)

    # and the zenith angle at the decay point
    zenith_decay[~ground] = get_decay_zenith_angle(decay_altitude[~ground], X0, ice)

    # calculate the distance for the other points
    distance[~ground] = get_distance_decay_to_detector(
        zenith_decay[~ground], view[~ground], decay_altitude[~ground], detector_altitude, ice
    )

    # and return the calculated distances
    return distance, zenith_decay


def geomag(
    station: np.ndarray, zenith: np.ndarray, azimuth: np.ndarray
) -> np.ndarray:
    
    '''
    with open('/dev/null', 'w') as devnull:
        oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)

        geomag = igrf('2022-10-12', glat=station[0], glon=station[1], alt_km=station[2])

        os.dup2(oldstdout_fno, 1)
        
        os.close(oldstdout_fno)
   ''' 

    geomag = igrf('2022-10-12', glat=station[0], glon=station[1], alt_km=station[2])
    
    mag = geomag.total.values[0]
    B = np.array([geomag.east.values, geomag.north.values, -geomag.down.values]).T
    V = np.array([np.sin(np.deg2rad(zenith))*np.cos(np.deg2rad(azimuth)), np.sin(np.deg2rad(zenith))*np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(zenith))]).T
    
    sinVB = np.linalg.norm(np.cross(V/np.linalg.norm(V, axis=1)[:,None], B/np.linalg.norm(B, axis=1)), axis=1)

    return mag, sinVB

