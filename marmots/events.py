"""
This file contains the parameters of the mystery events for A1/A3.
"""
import numpy as np

# this array contains the two A1/A3 mystery events
# flight: int
# elevation float [degrees]
# azimuth: float [degrees]
# altitude: float [m]
# efield: float [mV]
# ANITA-4 event is currently a placeholder
mystery = np.asarray(
    [
        (1, 3985267, -27.4, -159.6, -81.39856, 129.01626, 35029, 0.77, 1167266000),
        (3, 15717147, -35.0, -61.4, -82.6559, 17.2842, 35861, 1.1, 1419064402),
        (4, 9999999, -30, 0, -90, 0, 35000, 1.0, 1549064402),
    ],
    dtype=[
        ("flight", int),
        ("id", int),
        ("elevation", float),
        ("azimuth", float),
        ("latitude", float),
        ("longitude", float),
        ("altitude", float),
        ("efield", float),
        ("time", float),
    ],
)


def from_flight(flight: int) -> np.ndarray:
    """
    Given a flight number, return the mystery event from that flight.

    Parameters
    ----------
    flight: int
        The ANITA flight number.

    Returns
    -------
    event: np.ndarray
        A numpy array containing the parameters for that event.
    """
    return mystery[mystery["flight"] == flight]
