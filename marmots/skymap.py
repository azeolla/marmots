from typing import Optional, Tuple, Union

import astropy.coordinates as coordinates
import astropy.units as units
import attr
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from astropy.time import Time
from mpl_toolkits.axes_grid1 import make_axes_locatable

import poinsseta.auger as auger
import poinsseta.effective_area as eff
import poinsseta.flightpath as flightpath

from scipy.ndimage import gaussian_filter
#from sklearn import preprocessing


@attr.s
class Skymap:
    """
    Store the results of a Skymap calculation.
    """

    # the total number of trials thrown for this result
    N: Union[np.ndarray, int] = attr.ib()

    # the right ascensions (degrees) where the skymap was evaluated
    ra: np.ndarray = attr.ib()

    # the declination (degrees) where the skymap was evaluated
    dec: np.ndarray = attr.ib()

    # and the skymap data
    data: np.ndarray = attr.ib()

    # allow two results be added
    def __add__(self, other: "Skymap") -> "Skymap":
        """
        Add two skymaps together. This implements
        the sum of the two skymaps.

        The skymaps must have been
        sampled at the same RA and Dec.

        Parameters
        ----------
        other: Skymap
            Another skymap.

        """

        # check that the RA's are the same
        if not np.isclose(self.ra, other.ra).all():
            raise ValueError(("Skymap must have been " "evaluated at the same RA's"))

        # check that the dec's are the same
        if not np.isclose(self.dec, other.dec).all():
            raise ValueError(
                ("Skymap must have been " "evaluated at the same declination")
            )

        # and add the total number of trials
        N = self.N + other.N

        # and add the other quantities
        ra = self.ra
        dec = self.dec
        data = self.data + other.data

        # and create a new AcceptanceResult
        return Skymap(N, ra, dec, data)

    def plot(
        self,
        prototype: Optional[int] = None,
        # label: str = r"Normalized Effective Area",
        label: str = r"Effective Area [km$^2$]",
        title: str = "Instantaneous Effective Area",
        auger_fov: Optional[str] = None,
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
        """
        Plot a skymap of a given quantity at a range
        of RA and Dec values.

        Parameters
        ----------
        prototype: int
            If provided, overlay the location of the prototype ME's.
        label: str
            The label to use for the colorbar.
        title: str
            The title to use for the plot
        auger: Optional[str]
            If "ES" or "DG", overlay the Auger FoV on top of the skymap.

        Returns
        -------
        fig, ax: Tuple[matplotlib.figure.Figure,
                    matplotlib.axes.Axes]
        """

        # and now we plot the skymap
        fig, ax = plt.subplots()

        # the location of our mesh points
        RA, DEC = np.meshgrid(self.ra, self.dec)

        # if we don't have a masked array
        if not isinstance(self.data, ma.MaskedArray):
            skymap = ma.masked_less_equal(self.data, 0)

        # fill in the skymap
        skymap = skymap.filled(0.1 * skymap.min() + 1e-16)

        # if we didn't get any events
        if skymap.sum() == 0:
            raise ValueError("Skymap contains no passing events (zero effective area)")

        skymap = gaussian_filter(skymap, sigma=1)

        # normalize the data
        # Z_norm = preprocessing.normalize(Z)

        # plot the mesh
        im = ax.pcolormesh(
            RA,
            DEC,
            # Z.T,
            skymap.T,
            cmap="inferno",
            # norm=colors.LogNorm(vmin=Z_norm.min() + 1e-6, vmax=Z_norm.max()),
            norm=colors.LogNorm(vmin=skymap.min() + 1e-13, vmax=skymap.max()),
        )
        # create the colorbar axis
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # and add the colorbar
        cbar = plt.colorbar(im, cax=cax)  # , format=formatter)

        # and give it a label
        cbar.set_label(label, rotation=270, labelpad=15)

        # if we want to overlay auger
        if auger_fov is not None and prototype is not None:
            auger.add_fov(self, fig, ax, prototype, search=auger_fov)

        # and some labels
        ax.set(
            xlabel=r"Right Ascension [$^\circ$]",
            ylabel=r"Declination [$^\circ$]",
            title=title,

        )

        # and some limits
        ax.set_ylim([-90, 90])
        ax.set_xlim([np.min(self.ra), np.max(self.ra)])

        # and return the figure and the axes
        return fig, ax


def effective_area(
    Enu: float,
    elevations: np.ndarray,
    average: bool,
    altitude: float = 3.87553,
    prototype: int = 2018,
    maxview: float = np.radians(3.0),
    thickness: int = 0,
    nazimuth: int = 720,
    N: Union[np.ndarray, int] = 1_000_000,
    antennas: int = 4,
    gain: float = 6.0,
    minfreq: float = 30,
    maxfreq: float = 80,
    trigger_sigma: float = 5,
) -> Skymap:
    """
    Produce an (RA, declination) skymap of the geometric
    area of BEACON throughout a given prototype trial.

    This random chooses `npoints` along the flight path, and
    uses `ntrials` to evaluate the geometric area at `nazimuth`*360
    by `nelevation`*90 points on the sky.

    Parameters
    ----------
    prototype: int
        The BEACON prototype to simulate.
    average: bool
        Whether or not to take the average over time.
    N: int
        The number of MC trials at each flightpath point.

    Returns
    -------
    ra, dec: np.ndarray, np.ndarray
        The RA and dec values that the skymap is sampled at [deg].
    skymap: np.ndarray
        The sampled geometric area at each bin [km^2].
    """

    # load the correct flight path
    path = flightpath.load_prototype(prototype)

    # the number of steps in elevation and azimuth
    nelevation = elevations.size

    # and the corresponding resolutions
    dazimuth = 360.0 / nazimuth
    delevation = np.abs(elevations.max() - elevations.min()) / nelevation

    # create the array of azimuths
    azimuths = np.arange(-180.0, 180.0 + dazimuth, dazimuth)

    # and mesh these
    alt = np.repeat(elevations, azimuths.size)
    az = np.tile(azimuths, elevations.size)

    # the R/A and dec that we use to fill the skymap
    ra = np.linspace(0.0, 360.0, nazimuth)
    dec = np.arange(-90.0, 90.0 + delevation, 1.05 * delevation)

    # and the array to store the geometric area
    skymap = np.zeros((ra.size, dec.size))

    # choose random points along the path
    # locations = np.random.randint(path.latitude.shape[0], size=npoints)

    # choose the first point
    locations = [0]

    # the location of BEACON at each of these points
    beacon = coordinates.EarthLocation(
        lat=path.latitude[locations] * units.deg,
        lon=path.longitude[locations] * units.deg,
        height=path.altitude[locations] * units.m,
    )

    # and get the time of of each prototype location
    time = Time(path.realTime[locations], format="unix")

    # construct the array of AltAz locations
    altaz = coordinates.AltAz(
        alt=alt * units.deg, az=az * units.deg, obstime=time[0], location=beacon[0]
    )

    # and put these in RA/DEC
    skycoord = altaz.transform_to(coordinates.ICRS)

    # calculate the effective area
    Aeff = eff.calculate(
        Enu,
        np.radians(elevations),
        altitude,
        prototype,
        maxview,
        thickness,
        N,
        antennas,
        gain,
        minfreq,
        maxfreq,
        trigger_sigma,
    )

    # find the indices into the skymap
    ira = np.round(np.interp(skycoord.ra.value, ra, np.arange(ra.size)))
    idec = np.round(np.interp(skycoord.dec.value, dec, np.arange(dec.size)))

    # and make sure they are int's
    ira = ira.astype(int)
    idec = idec.astype(int)

    # it's possible to get some weird aliasing issues so let's apply
    # a basic bilinear anti-aliasing filter using np.random.normal
    # to blur the samples in each direction by one or two pixels.
    ira += +np.random.normal(loc=0, scale=1, size=ira.size).astype(int)
    idec += np.random.normal(loc=0, scale=1, size=idec.size).astype(int)

    # make sure ira and idec are clipped to the right range.
    ira = np.clip(ira, 0, ra.size - 1)
    idec = np.clip(idec, 0, dec.size - 1)

    # extract the effective area at each elevation angle
    effective = Aeff.effective_area

    # and add the points to the skymap
    skymap[ira, idec] += np.repeat(effective, azimuths.size)

    # for a skymap of the average effective area
    # slide the skymap along the RA and sum, then take the average.
    if average:
        for i in range(1, ra.size):
            ira2 = ira + i
            for j in range(0, ira2.size):
                if ira2[j] > (ra.size - 1):
                    ira2[j] = ira2[j] - (ra.size)
            skymap[ira2, idec] += np.repeat(effective, azimuths.size)
        skymap /= ra.size

    # and return the required quantities
    return Skymap(N, ra, dec, skymap)
