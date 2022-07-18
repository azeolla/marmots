"""
This module provides implementations of the various BEACON triggers.
"""
import numpy as np

import poinsseta.antenna as antenna


def trigger_level(
    freqs: np.ndarray, prototype: int, antennas: int, trigger_sigma: float,
) -> float:
    """
    Return the trigger level in V for a given BEACON payload.

    Parameters
    ----------
    freqs: np.ndarray
         The frequencies to calculate at (in MHz).
    prototype: int
        The BEACON payload to simulate.
    antennas: int
        The number of antennas.
    trigger_sigma: float
        The number of sigma for the trigger threshold.

    Returns
    -------
    Eftrig: float
        The trigger-threshold in V.
    """
    if prototype == 2018:
        return trigger_sigma * np.sqrt(
            np.sum(antenna.noise_voltage(freqs, prototype, antennas))
        )
    elif prototype == 2019:
        return trigger_sigma * np.sqrt(
            np.sum(antenna.noise_voltage(freqs, prototype, antennas))
        )
    else:
        raise ValueError(f"{prototype} is not a valid BEACON prototype for poinsseta.")
