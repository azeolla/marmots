"""
This module provides the high-level event loop to calculate
the tau point source effective area.
"""
from typing import Any, Union

import numpy as np

# import marmots.events as events
import marmots.geometry as geometry

import astropy.coordinates as coordinates
import astropy.units as u


def calculate(
    source: coordinates.SkyCoord,
    altaz: Any,
    beacon: coordinates.EarthLocation,
    maxview: float = np.radians(3.0),
    N: Union[np.ndarray, int] = 1_000_000,
) -> float:
    """
    Calculate the effective area of BEACON to a point source
    tau flux.

    Parameters
    ----------
    Enu: float
        The energy of the neutrino that is incident.
    elev: np.ndarray
       The elevation angle (in radians) to calculate the effective area at.
    altitude: float
       The altitude of BEACON (in km) for payload angles.
    prototype: int
        The prototype number for this BEACON trial.
    maxview: float
        The maximum view angle (in degrees).
    N: Union[int, np.ndarray]
        The number of trials to use for geometric area.
    antennas: int
        The number of antennas.
    freqs: numpy array
        The frequencies at which to calculate the electric field (in MHz).
    trigger_sigma: float
        The number of sigma for the trigger threshold.

    Returns
    -------
    Aeff: EffectiveArea
        A collection of effective area components across elevation.
    """

    # compute the geometric area at the desired elevation angles
    Ag = geometry.geometric_area(
        beacon, source, altaz, maxview, N
    )

    # and save the various effective area coefficients at these angles
    geometric = (Ag.area * np.sum(Ag.dot)) / N
    
    # and now return the computed parameters
    return geometric
