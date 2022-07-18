"""
Various plots, data files, and methods relating
to comparisons with the Pierre Auger Observatory.
"""
from typing import Any

import astropy.coordinates as coordinates
import astropy.units as units
import matplotlib
import numpy as np
import numpy.ma as ma

# from astropy.time import Time

# import poinsseta.events as events


def add_fov(
    skymap: Any,
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes._axes.Axes,
    prototype: int,
    search: str = "ES",
    **kwargs: Any,
) -> None:
    """
    Add the Auger \nu_\tau FoV to an existing Skymap plot.

    If search == "ES", plot the Earth skimming acceptance.
    If search == "DG", plot the combined downgoing acceptance.
    If search == "DGL", plot the downgoing "low" acceptance.
    If search == "DGH", plot the downgoing "high" acceptance.

    Parameters
    ----------
    skymap: Skymap
        The skymap to add the FoV to.
    fig: matplotlib.figure.Figure
        The figure to add the FoV to.
    ax:
        The matplotlib axes to add the figure to.
    prototype: int
        The prototype to plot.
    search: str
        The type of search to plot.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `search` it not in ["ES", "DG"]
    """

    # get the mystery event for this prototype
    # event = events.from_prototype(prototype)

    # and get the time for this event
    # time = Time(np.asarray(event["time"]), format="unix")
    time = 1

    # the elevation angles we sample
    if search == "ES":
        elevations = np.arange(0.0, -5.0, 0.2)
    elif search == "DG":
        elevations = np.arange(0.0, 30.0, 0.2)
    elif search == "DGL":
        elevations = np.arange(15.0, 30.0, 0.2)
    elif search == "DGH":
        elevations = np.arange(0.0, 15.0, 0.2)
    else:
        raise ValueError(f"{search} is not a valid Auger search type.")

    # the number of steps in elevation and azimuth
    nelevation = elevations.size

    # the elevation resolution
    delevation = np.abs(elevations.max() - elevations.min()) / nelevation

    # construct the location of Auger
    auger = coordinates.EarthLocation(
        lat=-35.4634 * units.deg, lon=-69.5848 * units.deg, height=1400 * units.m,
    )

    # the number of azimuth sampling points we use
    nazimuth = 1080

    # and the resolution in azimuth
    dazimuth = 360.0 / nazimuth

    # create the array of azimuths
    azimuths = np.arange(-180.0, 180.0 + dazimuth, dazimuth)

    # and mesh these
    alt = np.repeat(elevations, azimuths.size)
    az = np.tile(azimuths, elevations.size)

    # the R/A and dec that we use to fill the skymap
    ra = np.linspace(0.0, 360.0, nazimuth)
    dec = np.arange(-90.0, 90.0 + delevation, 1.05 * delevation)

    # construct the array of AltAz locations
    altaz = coordinates.AltAz(
        alt=alt * units.deg, az=az * units.deg, obstime=time, location=auger
    )

    # and put these in RA/DEC
    skycoord = altaz.transform_to(coordinates.ICRS)

    # find the indices into the skymap
    ira = np.round(np.interp(skycoord.ra.value, ra, np.arange(ra.size)))
    idec = np.round(np.interp(skycoord.dec.value, dec, np.arange(dec.size)))

    # and make sure they are int's
    ira = ira.astype(int)
    idec = idec.astype(int)

    # make sure ira and idec are clipped to the right range.
    ira = np.clip(ira, 0, ra.size - 1)
    idec = np.clip(idec, 0, dec.size - 1)

    # allocate new memory for the FoV
    fov = np.zeros((ra.size, dec.size))

    # and add the points to the skymap
    fov[ira, idec] = 1.0

    # mask the skymap where no values where found
    fov = ma.masked_less_equal(fov, 0)

    # the location of our mesh points
    RA, DEC = np.meshgrid(ra, dec)

    # and add it to the plot
    ax.pcolormesh(RA, DEC, fov.T, cmap="gray", vmin=0.9, vmax=1.0, alpha=0.6)

    # return the FoV
    return fov
