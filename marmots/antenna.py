"""
This module parameterizes the response of BEACON antennas to electric fields.
"""
import numpy as np

import poinsseta.sky as sky
from poinsseta.constants import Z_0, c_km, k_b

__all__ = [
    "noise_voltage",
    "voltage_from_field",
    "get_Tsys",
    "get_Tground",
]

# we assume that our antennas are nominally 50-ohms
Z_A = 50.0

# and that they see a 50 ohm load
Z_L = 50.0

# we currently assume that the antenna sees 50% sky and 50% ice
sky_frac = 0.5


def voltage_from_field(
    Epeak: np.ndarray, freqs: np.ndarray, antennas: int, gain: float,
) -> np.ndarray:
    """
    Given a peak electric field (in V/m), calculate the voltage seen
    at the load of the BEACON antenna.

    See any RF or antenna textbook for a derivation of this.

    Parameters
    ----------
    Epeak: np.ndarray
        The peak-electric field (in V/m).
    freqs: np.ndarray
        The frequencies (in MHz) at which to evaluate.
    antennas: int
        The number of antennas.
    gain: float
        The peak gain (in dBi).

    Returns
    -------
    voltage; np.ndarray
        The voltage seen at the load of the antenna.
    """

    # calculate the linear gain - `gain` must be power gain.
    G = np.power(10.0, gain / 10)

    # this is the voltage seen by the antenna
    V_A = (
        2
        * Epeak
        * antennas
        * (1e3 * c_km / (1e6 * freqs))
        * np.sqrt((Z_A / Z_0) * G / (4 * np.pi))
    )

    # and put this through the voltage divider to get the
    # voltage seen at the load
    V_L = V_A * Z_L / (Z_A + Z_L)

    # and we are done
    return V_L


def noise_voltage(
    freqs: np.ndarray, prototype: int = 2018, antennas: int = 4,
) -> np.ndarray:
    """
    Return the noise voltage (in V) at given frequencies.

    Parameters
    ----------
    freqs: np.ndarray
        The frequencies to calculate at (in MHz).
    prototype: int
        The BEACON prototype to use.
    antennas: int
        The number of antennas.

    Returns
    -------
    noise: np.ndarray
        The noise voltages at each frequency (in V).
    """

    # get the sky noise temperature in Kelvin
    Tsky = sky.power_to_temperature(freqs, sky.galactic_noise(freqs))

    # and the system noise temperature in Kelvin
    Tsys = get_Tsys(freqs, prototype)

    # and the ground temperature in Kelvin
    Tground = get_Tground(freqs)

    # combine to make the total noise temperature
    Tcomb = Tsys + Tground * (1.0 - sky_frac) + Tsky * sky_frac

    # get the bandwidth at each frequency step (Hz)
    bw = 1e6 * (freqs[1] - freqs[0])

    # and now convert this into a combined noise voltage
    Vn = np.sqrt(antennas * Tcomb * bw * k_b * Z_L)

    # still needs to be summed, then square-rooted
    return Vn


def get_Tground(freqs: np.ndarray) -> np.ndarray:
    """
    Get the ground temperature (in K) at given frequencies.

    The apparent temperature of the ground is not just
    the physical temperature as the sky background
    reflects off the ground into the antenna.

    Parameters
    ----------
    freqs: np.ndarray
        The frequencies in MHz.

    Returns
    -------
    Tsys: np.ndarray
        The system temperature (in Kelvin).
    """

    # we currently assume a constant temperature
    return 290.0 * np.ones_like(freqs)


def get_Tsys(freqs: np.ndarray, prototype: int = 2018) -> np.ndarray:
    """
    Get the system temperature at a set of frequencies
    for a specific BEACON prototype.

    Parameters
    ----------
    freqs: np.ndarray
        The frequencies in MHz.
    prototype: int
        The BEACON prototype to use.

    Returns
    -------
    Tsys: np.ndarray
        The system temperature (in Kelvin).
    """

    # we currently assume a constant temperature
    return 140.0 * np.ones_like(freqs)
