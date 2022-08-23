"""
This class provides access to the tau exit lookup tables.
"""
import os.path as path
from typing import Any, Tuple

import attr
import numpy as np
import numpy.ma as ma
from numba import njit

from marmots import data_directory


@attr.s
class TauExitLUT:
    """
    This class allows sampling from the tau LUT's.

    Creating a TauExitLUT without arguments defaults to Enu=1e18,
    and 2.0km ice thickness

    >>> t = TauExitLUT()

    To change the energy or ice thickness, pass the keyword parameters
    to the constructor.

    >>> t = TauExitLUT(energy=3e19, thickness=1.0)

    The properties of the LUT can be accessed as class members:

    >>> t.exit_energies   # the array of exitting tau energies for each sim
    >>> t.exit_theta      # the exit angle (degrees) for each sim.
    >>> t.Pexit           # the calculated exit probability for each sim.

    To interpolate an exit probability, use the get method.

    >>> t.get_pexit(20.5) # the exit probability at 20.5d exit angle.

    To randomly sample an exit probability and energy, call the class

    >>> Pexit, energy = t(20.5) # sampled the LUT at 20.5d exit angle.
    >>> Pexit, energy = t(20.5*np.ones(100)) # array oriented version.

    To create a LUT directly from a filename, use the filename keyword.

    >>> t = TauExitLUT(filename="2.0km_ice_midCS_stdEL/LUT_3e+16_eV")

    If you want to load the LUTs from a different directory, use the
    second (`lutdir`) argument to the constructor.

    >>> t = TauExitLUT(filename="2.0km_ice_midCS_stdEL/LUT_3e+16_eV",
                       lutdir="my_other_LUT_dir")
    """

    # the true energy of the desired LUT
    energy: float = attr.ib(default=1e18)

    # and the ice thickness of the desired LUT
    thickness: float = attr.ib(default=2.0)

    # the (optional) filename of the parameterization
    filename: str = attr.ib(default=None, kw_only=True)

    # the directory where we store parameterizations
    lutdir: str = attr.ib(default=path.join(data_directory, "tauexit"), kw_only=True)

    # the number of samples in the quantile function
    Nsamples: int = attr.ib(default=100, kw_only=True)

    def get_pexit(self, exittheta: np.ndarray) -> np.ndarray:
        """
        Calculate the exit probability at each exit angle (in degrees)
        in exittheta using linear interpolation.

        Parameters
        ----------
        exitheta: np.ndarray
            The desired exit angles (in degrees).

        Returns
        -------
        Pexit: np.ndarray
            The calculated exit probabilities.
        """
        return np.interp(90 - exittheta, 90 - self.exit_theta, self.Pexit)

    def __call__(self, exittheta: np.ndarray) -> Tuple[np.ndarray, ma.array]:
        """
        Sample a random exit energy of a tau at a given exit angle.
        This also returns the exit probability associated with that exit angle.

        If no tau's exitted the Earth in this LUT at the given exit angle,
        the corresponding array entry will be masked.

        Parameters
        ----------
        exittheta: np.ndarray
            An array of tau exit angles (in degrees).

        Returns
        -------
        Pexit: ma.masked_array
            The exit probability associated with each exit angle.
        energies: ma.masked_array
            A randomly sampled tau exit energy at each exit angle.
        """

        # check if this is a numpy array
        exittheta = np.atleast_1d(exittheta)

        # find the exit probability at each exit theta
        Pexit = self.get_pexit(exittheta)

        # find the closest index in angle
        nearest = np.abs(exittheta[:, None] - self.exit_theta[None, :]).argmin(axis=-1)

        # and make sure index is within a valid range
        idxs = np.clip(nearest, 0, self.exit_theta.size - 1)

        # and get the energies from the quantile function
        energies = interp(
            exittheta,
            self.exit_theta,
            self.quantiles,
            self.energy_quantiles,
            idxs,
            self.nexit,
        )

        # and we mask any values that are zero
        energies = ma.masked_equal(energies, 0.0)

        # and return the exit probability and the energies
        return Pexit, np.power(10.0, energies)

    def __attrs_post_init__(self) -> None:
        """
        Called at the end of __init__. This opens the LUT,
        saves the needed arrays into the class, and compute some
        basic properties.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # if filename is None, construct the filename based on the arguments
        if not self.filename:
            directory = f"{self.thickness:.1f}km_ice_midCS_stdEL"
            fname = f"LUT_{self.energy:.0}_eV"
            self.filename = path.join(directory, fname)
        else:  # otherwise parse out the energies
            self.thickness = float(self.filename[0:2])
            self.energy = float(self.filename.split("_")[-2])

        # open the LUT - need to manually specify encoding and allow pickling
        # since .npz files were produced using Python 2 (yuck).
        with np.load(
            path.join(self.lutdir, self.filename + ".npz"),
            encoding="bytes",
            allow_pickle=True,
        ) as lut:

            # the exit energies for this simulation
            exit_energies = lut["data_array"]

            # the number of bins in the quantile function
            self.quantiles = np.linspace(0.0, 1.0, self.Nsamples)

            # create an array with the number of exitting energies per sim
            self.nexit = np.zeros(len(exit_energies))

            # the number of angular bins in the lut
            Nbins: int = len(lut["data_array"])

            # create the array to store the quantiles
            self.energy_quantiles = np.zeros((Nbins, self.Nsamples))

            # and loop over all energy results
            for i in np.arange(len(lut["data_array"])):

                # save the number of exitting energies per sim
                self.nexit[i] = exit_energies[i].size

                # if we didn't get any exit energies, then set the quantiles to zero
                if self.nexit[i] == 0:
                    self.energy_quantiles[i, :] = np.ones_like(self.quantiles)
                else:
                    # otherwise, compute the quantiles of the exit energies
                    self.energy_quantiles[i, :] = np.quantile(
                        np.power(10.0, exit_energies[i]), self.quantiles
                    )

            # the exit theta of each simulation
            self.exit_theta = 90 + lut["th_exit_array"]

            # and the number of tau's thrown in each simulation
            self.nthrown = lut["num_sim"]

        # and now calculate the exit probability at each angle
        self.Pexit = self.nexit / self.nthrown

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

        if name:
            # check if file doesn't exist
            if not path.exists(path.join(self.lutdir, name + ".npz")):
                raise ValueError(f"{name} cannot be found in {self.lutdir}")


@njit
def interp(
    theta: np.ndarray,
    exit_theta: np.ndarray,
    quantiles: np.ndarray,
    energy_quantiles: np.ndarray,
    idxs: np.ndarray,
    nexit: np.ndarray,
) -> np.ndarray:
    """
    Interpolate into the quantile for each bin.

    Parameters
    ----------
    theta: np.ndarray
        The into exit_theta to interpolate.
    exit_theta: np.ndarray
        The angular bins in the LUT.
    quantiles: np.ndarray
        The range of quantiles to sample from.
    energy_quantiles: np.ndarray
        The actual quantiles in energy.
    idxs: np.ndarray
        The indexes into the first axis of energy_quantiles
    nexit: np.ndarray
        The number of exitting taus in each LUT bin.
    Returns
    -------
    energy_samples: np.ndarray
        Energies sampled from the CDF.
    """

    # choose the random uniform fraction into the quantile
    u = np.random.random(size=theta.size)
    ulow = np.random.random(size=theta.size)
    uhigh = np.random.random(size=theta.size)

    # the output array
    energies = np.zeros(u.shape[-1], dtype=np.float64)

    # loop over the provided indices
    for i in np.arange(u.shape[-1]):

        # get current index into the quantiles
        qidx = idxs[i]

        # if there are no tau's in the central bin, then set a zero energy
        if nexit[qidx] == 0:
            energies[i] = 0.0

        # otherwise, we have some taus to work with

        # sample from the lower angular bin
        if qidx >= 1 and nexit[qidx - 1] != 0:
            low = np.interp(ulow[i], quantiles, energy_quantiles[qidx - 1, :])

        # sample from the closest angular bin
        if nexit[qidx] != 0:
            mid = np.interp(u[i], quantiles, energy_quantiles[qidx, :])

        # sample from the upper angular bin
        if qidx < idxs.shape[0] - 1 and nexit[qidx + 1] != 0:
            high = np.interp(uhigh[i], quantiles, energy_quantiles[qidx + 1, :])

        # we are at the left-most edge of the LUT
        if qidx == 0:

            # if we have a neighboring bin, then interpolate
            if nexit[qidx + 1] != 0:
                energies[i] = np.interp(
                    theta[i], exit_theta[qidx : qidx + 2], np.asarray([mid, high])
                )
            else:  # just return this one bin
                energies[i] = np.interp(u[i], quantiles, energy_quantiles[qidx, :])

        # we are at the right-most edge of the bin
        elif qidx == idxs.shape[0] - 1:
            if nexit[qidx - 1] != 0:
                energies[i] = np.interp(
                    theta[i], exit_theta[qidx - 1 : qidx + 1], np.asarray([low, mid])
                )
            else:  # just sample this bin
                energies[i] = np.interp(u[i], quantiles, energy_quantiles[qidx, :])
        else:  # we are in the middle of our lut so do some interpolation

            # we now explicitly list out all our cases for various tau's
            # this interpolated sampling is good when the number of tau's
            # in any bin is small but is not really needed if the LUT's
            # are sufficiently well sampled (high number of nthrown)

            # tau's in all three bins
            if nexit[qidx - 1] != 0 and nexit[qidx] != 0 and nexit[qidx + 1] != 0:
                energies[i] = np.interp(
                    theta[i],
                    exit_theta[qidx - 1 : qidx + 2],
                    np.asarray([low, mid, high]),
                )

            # no tau's in the low bin and no tau's in the high bin
            # but we have tau's in the center bin
            elif (
                nexit[qidx - 1] == 0 and nexit[qidx + 1] == 0
            ):  # we got no tau's in the low bin
                energies[i] = np.interp(u[i], quantiles, energy_quantiles[qidx, :])

            # tau's in the mid and high-bins
            elif nexit[qidx] != 0 and nexit[qidx + 1] != 0:
                energies[i] = np.interp(
                    theta[i], exit_theta[qidx : qidx + 2], np.asarray([mid, high])
                )

            # tau's in the low and high bins
            elif nexit[qidx - 1] != 0 and nexit[qidx + 1] != 0:
                energies[i] = np.interp(
                    theta[i],
                    exit_theta[qidx - 1 : qidx + 2 : 2],
                    np.asarray([low, high]),
                )
            # tau's in the low and mid bins
            elif nexit[qidx - 1] != 0 and nexit[qidx] != 0:
                energies[i] = np.interp(
                    theta[i], exit_theta[qidx - 1 : qidx + 1], np.asarray([low, mid]),
                )

    # we did all the interpolation in eV units
    # but we want to return log10(eV) to the caller
    energies = np.log10(energies)

    # and return the output energies
    return energies
