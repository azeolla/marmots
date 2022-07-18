"""
Evaluate properties of tau decays.
"""
import numpy as np


def probability(decay_length: np.ndarray, dbeacon: np.ndarray) -> np.ndarray:
    """
    Return the probability that each tau (eV) decays
    before traveling `dbeacon` (km)`

    Parameter
    ---------
    Etau: np.ndarray
        The energy of each tau in eV.
    dbeacon: np.ndarray
        The distance from exit to BEACON (in km).

    Returns
    -------
    Pdecay: np.ndarray
        The decay probability at each tau energy.
    """
    return 1.0 - np.exp(-dbeacon / decay_length)
