"""
This class provides access to the parametrized electric field datafiles.
"""
import os.path as path
from typing import Any, Tuple

import attr
import numpy as np
from interpolation.splines import CGrid, eval_linear
from numba import njit
from scipy.interpolate import interpn

import poinsseta.antenna as antenna
from poinsseta import data_directory
from poinsseta.constants import Re


def get_distance_decay_to_detector(
    trials: np.ndarray,
    axis: np.ndarray,
    decay_length: np.ndarray,
    beacon: np.ndarray,
) -> np.ndarray:
    # calculates the distance from the decay point to the detector
    
    decay_point = trials + axis * decay_length[:, None]
    d = np.linalg.norm(beacon - decay_point, axis=1)
    return d



@njit
def eval(
    grid: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    values: np.ndarray,
    freqs: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
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
    d: np.ndarray
        The decay altitudes to interpolate at (km).
    z: np.ndarray
        The zenith angles to interpolate at (degrees).
    v: np.ndarray
        The view to interpolate at (degrees).
    Returns
    -------
    Efield: np.ndarray
       The electric field interpolated at each (f, d, z, v).
    """
    # allocate the output array
    out = np.zeros((freqs.shape[-1], z.shape[-1]), dtype=np.float64)

    # loop over the array
    for i in np.arange(freqs.shape[-1]):
        # perform the interpolation
        out[i, :] = eval_linear(
            grid, values, np.column_stack((np.repeat(freqs[i], z.shape[-1]), d, z, v))
        )
    return out


@attr.s
class EFieldParam(object):
    """
    Load and sample the included BEACON E-field parameterization files.
    """

    # the filename of the parameterization
    filename: str = attr.ib(default="interpolator_efields_4.0km")

    # the LUT of distances to the detector in each sim
    Dsim: np.ndarray = attr.ib(default=None)

    # the directory where we store parameterizations
    param_dir = path.join(data_directory, "beacon")

    def __call__(
        self,
        view: np.ndarray,
        zenith: np.ndarray,
        decay_altitude: np.ndarray,
        decay_length: np.ndarray,
        axis: np.ndarray,
        trials: np.ndarray,
        beacon: np.ndarray,
        freqs: np.ndarray,
        tau_energy: np.ndarray,
        altitude: float,
        gain: float,
        antennas: int,
    ) -> np.ndarray:
        """
        Evaluate the peak electric field from this parameterization at
        a given off-axis angle (in degrees) given the zenith angle
        (in degrees), decay altitude (in km) of the tau, and the frequency (in MHz).
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
        tau_energy: np.ndarray
            The energy of the tau
        detector_altitude:
            The altitude we want the effective areas at.
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
        zenith = np.clip(exit_zenith, 55.0, 89.0)
        view = np.clip(view, 0.04, 3.16)
        decay = np.clip(decay_altitude, 0, self.sim_altitude - 0.5)

        # interpolate to find ZHAireS' distances at the zenith angles
        # and decay altitudes that we want.
        r_zhaires_tau_shower = interpn(
            (self.zenith_list, self.decay_list),
            self.Dsim,
            (zenith, decay),
            bounds_error=False,
            fill_value=None,
        )

        # calculate the distance from decay to detector in Poinsseta
        distance_decay_km = distance_decay_detector(
            trials, axis, decay_length, beacon
        )

        # calculate the voltage at each frequency
        df = 10.0
        voltage = antenna.voltage_from_field(
            eval(self.grid, self.values, freqs, decay, zenith, view),
            (freqs + df / 2.0).reshape(-1, 1),
            antennas,
            gain,
        )

        # account for ZHAIReS sims only extending to 3.16 deg
        voltage[:, view > 3.16] *= np.exp(
            -((view[view > 3.16] - 0.0) ** 2) / (2 * 3.16) ** 2
        )

        # distance correction (ZHAireS distance over Poinsseta distance)
        voltage *= r_zhaires_tau_shower / distance_decay_km

        # energy scaling
        voltage *= tau_energy / self.e_zhaires_tau_shower

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

        # mesh the loaded decay altitudes and zenith angles
        Da, Za = np.meshgrid(self.decay_list, self.zenith_list)

        # calculate the distance to the detector in each sim
        Dsim = distance_decay_detector(
            self.zhaires_sim_icethick, Da.flatten(), self.sim_altitude, Za.flatten()
        )

        # reshape the array to the appropriate size and save
        self.Dsim = Dsim.reshape((self.zenith_list.size, self.decay_list.size))

    def load_file(self) -> None:
        """
        Load the parameterization file and store it into the class.
        """
        # load the data file
        interp_file = np.load(
            path.join(self.param_dir, self.filename + ".npz"),
            allow_pickle=True,
            encoding="bytes",
        )

        efield_interpolator_list = interp_file["efield_interpolator_list"][()]

        # ZHAireS sim parameters
        self.zhaires_sim_icethick = 0.0
        self.sim_altitude = float(self.filename[21:][:-2])
        self.e_zhaires_tau_shower = 1e17

        self.zenith_list = efield_interpolator_list[0].points[0 : 9 * 80 : 80, 0]
        view_list = efield_interpolator_list[0].points[0:80:1, 2]
        self.decay_list = efield_interpolator_list[0].points[0::720, 1]
        freqs = np.arange(10.0, 1610.0, 10.0)

        self.grid = CGrid(freqs, self.decay_list, self.zenith_list, view_list)
        self.values = np.zeros(
            (len(freqs), len(self.decay_list), len(self.zenith_list), len(view_list))
        )

        # values has format (frequency, decay, zenith, view)
        for i in np.arange(freqs.size):
            self.values[i, ...] = (
                efield_interpolator_list[i]
                .values[:, 0]
                .reshape((self.decay_list.size, self.zenith_list.size, view_list.size))
            )

    @filename.validator
    def check(self, _: Any, name: str) -> None:
        """
        Check that the filename exists in the data directory.

        This raises a ValueError if the file cannot be found.

        Parameters
        ----------
        _: str
            The name of the attribute to validate [not used]
        name: str
            The value of the attribute to validate.

        Returns
        -------
        None
        """

        # check if file doesn't exist
        if not path.exists(path.join(self.param_dir, name + ".npz")):
            raise ValueError(f"{name}.npz cannot be found in {self.param_dir}")
