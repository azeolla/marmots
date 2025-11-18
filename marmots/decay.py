"""
Evaluate properties of tau decays.
"""
import numpy as np


def probability(decay_length: np.ndarray, dbeacon: np.ndarray) -> np.ndarray:
    """
    Return the probability that a tau with decay length `decay_length` decays
    before traveling `dbeacon` (km)`

    Parameter
    ---------
    decay_length: np.ndarray
        The decay length of the tau (in km).
    dbeacon: np.ndarray
        The distance from exit to BEACON (in km).

    Returns
    -------
    Pdecay: np.ndarray
        The decay probability at each tau energy.
    """
    return 1.0 - np.exp(-dbeacon / decay_length)
