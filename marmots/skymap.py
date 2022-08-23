from typing import Optional, Tuple, Union

import astropy.coordinates as coordinates
import astropy.units as u
from astropy.time import Time
import attr
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import healpy as hp
from p_tqdm import p_map

from mpl_toolkits.axes_grid1 import make_axes_locatable

import marmots.effective_area as eff
import marmots.flightpath as flightpath

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
    latitude: np.ndarray,
    longitude: np.ndarray,
    altitude: np.ndarray,
    maxview: float = np.radians(3.0),
    nside: int = 16,
    N: Union[np.ndarray, int] = 1_000_000,
    antennas: int = 4,
    freqs: np.ndarray = np.arange(30,80,10)+5,
    trigger_sigma: float = 5,
    num_cpus: int = 1,
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

    # the number of pixels in the healpy skymap, determined by nside
    npix = hp.nside2npix(nside)

    # the location of each pixel
    theta, phi = np.degrees(hp.pix2ang(nside=16, ipix = np.arange(npix)))

    dec = 90 - theta
    ra = phi

    # the location of the BEACON stations
    beacon = coordinates.EarthLocation(
        lat=latitude * u.deg,
        lon=longitude * u.deg,
        height=altitude * u.m,
    )

    time = Time('2022-7-18 12:00:00') - 4*u.hour

    def pix_loop(i):
        source = coordinates.SkyCoord(ra=ra[i]*u.degree, dec=dec[i]*u.degree, frame='icrs')
        altaz = source.transform_to(coordinates.AltAz(obstime=time,location=beacon))

        geo = source.transform_to(coordinates.ITRS(obstime=time))
        geo.representation_type = 'spherical'

        # calculate the effective area
        Aeff = eff.calculate(
            Enu,
            geo,
            altaz,
            beacon,
            maxview,
            N,
            antennas,
            freqs,
            trigger_sigma,
        )

        return Aeff

    effective_area = p_map(pix_loop, np.arange(npix), **{"num_cpus": num_cpus})

    # and return the required quantities
    return Skymap(N, ra, dec, skymap)



