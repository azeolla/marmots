"""
This modules provides various functions to calculate RF
noise properties of the sky.
"""
import numpy as np

from marmots.constants import c_km, k_b

__all__ = [
    "noise_temperature",
    "galactic_noise",
    "extragalactic_noise",
    "power_to_temperature",
]


def noise_temperature(freqs: np.ndarray) -> np.ndarray:
    """
    """

    # the combined sky noise
    noise = galactic_noise(freqs) + extragalactic_noise(freqs)

    # and convert it to temperature
    return power_to_temperature(freqs, noise)


def power_to_temperature(freqs: np.ndarray, power: np.ndarray) -> np.ndarray:
    """
    Convert noise power in W m^-2 Hz^-1 sr^-1 into a
    noise temperature in Kelvin.

    See any standard radio astronomy book or the Dulk 2001
    paper cited for galactic noise.

    Parameters
    ----------
    freqs: np.ndarray
       The frequencies of each noise sample (in MHz).
    power: np.ndarray
        Noise power in W m^-2 Hz^-1 sr^-1.

    Returns
    -------
    Tnoise: np.ndarray
       The corresponding noise temperature (in Kelvin)./
    """

    # compute the frequencies in Hz
    nu = freqs * 1e6

    # and the true speed of light
    c = c_km * 1e3

    # and we are done
    return (power / k_b) * (c * c / (2 * nu * nu))


def galactic_noise(freqs: np.ndarray) -> np.ndarray:
    """
    Calculate the galactic contribution to RF background noise (in
    W m^-2 Hz^-1 sr^-1) as function of frequency (in MHz).

    These parameterization is taken from (Dulk, 2001):
    https://www.aanda.org/articles/aa/full/2001/02/aads1858/aads1858.right.html

    Parameters
    ----------
    freqs: np.ndarray
        The frequencies (in MHz) to evaluate at.

    Returns
    -------
    noise: np.ndarray
        The galactic contribution to sky noise in W m^-2 Hz^-1 sr^-1.
    """

    # the scaling factor for the galactic contribution.
    Ig = 2.48e-20

    # and the tau-factor
    tau = 5 * np.power(freqs, -2.1)

    # and evaluate the galactic form
    return Ig * np.power(freqs, -0.52) * ((1 - np.exp(-tau)) / tau)


def extragalactic_noise(freqs: np.ndarray) -> np.ndarray:
    """
    Calculate the extragalactic contribution to RF background noise (in
    W m^-2 Hz^-1 sr^-1) as function of frequency (in MHz).

    These parameterization is taken from (Dulk, 2001):
    https://www.aanda.org/articles/aa/full/2001/02/aads1858/aads1858.right.html

    Parameters
    ----------
    freqs: np.ndarray
        The frequencies (in MHz) to evaluate at.

    Returns
    -------
    noise: np.ndarray
        The galactic contribution to sky noise in W m^-2 Hz^-1 sr^-1.
    """

    # the scaling factor for the galactic contribution.
    Ieg = 1.06e-20

    # and the tau-factors
    tau = 5 * np.power(freqs, -2.1)

    # and evaluate the galactic form
    return Ieg * np.power(freqs, -0.8) * np.exp(-tau)
