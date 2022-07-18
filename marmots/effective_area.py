"""
This module provides the high-level event loop to calculate
the tau point source effective area.
"""
import pickle
from typing import Any, Dict, List, Tuple, Union

import attr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from tqdm import tqdm

import poinsseta.antenna as antenna
import poinsseta.decay as decay

# import poinsseta.events as events
import poinsseta.geometry as geometry
import poinsseta.tauola as tauola
import poinsseta.trigger as trigger
from poinsseta.efield import EFieldParam
from poinsseta.tauexit import TauExitLUT


# we provide the results of the effective area calculation in a tuple
@attr.s
class EffectiveArea:
    # the total number of trials thrown for this result
    N0: np.ndarray = attr.ib()

    # the elevation angles at which the effective area is sampled
    elevation: np.ndarray = attr.ib()

    # the effective area in kilometers at each angle
    effective_area: np.ndarray = attr.ib()

    # the geometric area in kilometers at each angle
    geometric: np.ndarray = attr.ib()

    # the exit probability at each angle
    pexit: np.ndarray = attr.ib()

    # the decay probability at each angle
    pdecay: np.ndarray = attr.ib()

    # the trigger probability at each angle
    ptrigger: np.ndarray = attr.ib()

    # the effective area due to noise triggers alone
    Anoise: np.array = attr.ib()

    # the arguments used to construct this effective area
    args: Dict[str, Any] = attr.ib()

    # allow two results be added
    def __add__(self, other: "EffectiveArea") -> "EffectiveArea":
        """
        Add two effective areas together. This implements
        the average of two effective areas.

        These two effective areas must have been
        sampled at the same elevation angles.

        Parameters
        ----------
        other: EffectiveArea
            Another effective area calculation.

        """

        # check that the angles are the same size
        if not self.elevation.size == other.elevation.size:
            raise ValueError(
                (
                    "Effective areas must have been "
                    "evaluated at the same elevation angles"
                )
            )

        # check that the angles are the same
        if not np.isclose(self.elevation, other.elevation).all():
            raise ValueError(
                (
                    "Effective areas must have been "
                    "evaluated at the same elevation angles"
                )
            )

        # and add the total number of trials
        N0 = self.N0 + other.N0

        # check if the args are the same
        if self.args != other.args:
            msg = "Effective areas generated with different arguments!"
            msg += f"self: \n{self.args}\n"
            msg += f"other: \n{other.args}\n"
            raise ValueError(msg)

        # and average all the quantities together
        effective_area = 0.5 * (self.effective_area + other.effective_area)
        geometric = 0.5 * (self.geometric + other.geometric)
        pexit = 0.5 * (self.pexit + other.pexit)
        pdecay = 0.5 * (self.pdecay + other.pdecay)
        ptrigger = 0.5 * (self.ptrigger + other.ptrigger)
        Anoise = 0.5 * (self.Anoise + other.Anoise)

        # and create a new EffectiveArea
        return EffectiveArea(
            N0,
            self.elevation,
            effective_area,
            geometric,
            pexit,
            pdecay,
            ptrigger,
            Anoise,
            self.args,
        )

    def plot(
        self, noise: bool = True
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
        """
        Parameters
        ----------
        noise: bool
            if True, also plot the effective area due to noise.
        """

        # and let's create a test plot as we work
        fig, ax = plt.subplots(figsize=(8, 4))

        # and sample some colors from a colormap (avoid the last color)
        colors = plt.cm.inferno(np.linspace(0, 1.0, 5))

        # plot the geometric area
        ax.semilogy(
            self.elevation,
            self.geometric,
            label=r"$\langle A_g \rangle$",
            color=colors[1],
            lw=1.0,
        )

        # and the geometric area times the exit probability
        ax.semilogy(
            self.elevation,
            self.geometric * self.pexit,
            label=r"$\langle A_g \rangle\ \bar{P}_{\mathrm{exit}}$",
            color=colors[2],
            lw=1.0,
        )

        # and incorporate the decay probability
        ax.semilogy(
            self.elevation,
            self.geometric * self.pexit * self.pdecay,
            label=r"$\langle A_g\rangle\ \bar{P}_{\mathrm{exit}}"
            r" \bar{P}_{\mathrm{decay}}$",
            color=colors[3],
            lw=1.0,
        )

        # plot the factorized geometric area
        ax.semilogy(
            self.elevation,
            self.geometric * self.pexit * self.pdecay * self.ptrigger,
            label=(
                r"$\langle A_g  \rangle\ \bar{P}_{\mathrm{exit}} "
                r"\bar{P}_{\mathrm{decay}} \bar{P}_{\mathrm{trig}}$"
            ),
            color=colors[0],
            alpha=0.5,
            lw=1.0,
        )

        # and plot the true effective area
        ax.semilogy(
            self.elevation,
            self.effective_area,
            label=(
                r"$\langle A_g  P_{\mathrm{exit}} "
                r"P_{\mathrm{decay}} P_{\mathrm{trig}} \rangle$"
            ),
            color=colors[0],
            lw=1.0,
        )

        # decide if we want to plot the noise
        if noise:
            ax.semilogy(
                self.elevation, self.Anoise, color=colors[0], ls=":", lw=1.0,
            )

        # and some labels
        ax.set(
            xlabel=r"Payload Elevation Angle [$^\circ$]",
            ylabel=r"Effective Area [km$^2$]",
            xlim=[self.elevation.min(), self.elevation.max()],
            ylim=[1e-12, 1e4],
        )

        # we want every order of magnitude
        ax.yaxis.set_major_locator(mtick.LogLocator(base=10.0, numticks=20))

        # enable the legend
        plt.legend()

        # and return the figures and the axes
        return fig, ax


def calculate(
    Enu: float,
    elev: np.ndarray,
    altitude: float = 3.87553,
    prototype: int = 2018,
    maxview: float = np.radians(3.0),
    icethickness: int = 0,
    N: Union[np.ndarray, int] = 1_000_000,
    antennas: int = 4,
    gain: float = 6.0,
    minfreq: float = 30,
    maxfreq: float = 80,
    trigger_sigma: float = 5.0,
) -> EffectiveArea:
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
    icethickness: int
        The thickness of the ice (in km).
        We currently support 0, 1, 2, 3, 4.
    N: Union[int, np.ndarray]
        The number of trials to use for geometric area.
    antennas: int
        The number of antennas.
    gain: float
        The peak gain (in dBi).
    minfreq: float
        The minimum frequency (in MHz).
    maxfreq: float
        The maximum frequency (in MHz).
    trigger_sigma: float
        The number of sigma for the trigger threshold.

    Returns
    -------
    Aeff: EffectiveArea
        A collection of effective area components across elevation.
    """

    # we make sure that elevation is at least an array
    elev = np.atleast_1d(elev)

    # make N an array if it's not already
    if not isinstance(N, np.ndarray):
        N = N * np.ones(elev.size, dtype=int)

    # load the corresponding tau exit LUT
    tauexit = TauExitLUT(energy=Enu, thickness=icethickness)

    # load the field parameterization.
    altitudes = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 37.0])
    i_altitude = np.abs(altitudes - altitude).argmin()
    altitude_file = altitudes[i_altitude]

    efield_filename = "interpolator_efields_" + str(altitude_file) + "km"
    voltage = EFieldParam(filename=efield_filename)

    # arrays to store the output of the effective area at each elevation
    effective_area = np.zeros_like(elev)
    geometric = np.zeros_like(elev)
    pexit = np.zeros_like(elev)
    pdecay = np.zeros_like(elev)
    ptrigger = np.zeros_like(elev)
    Anoise = np.zeros_like(elev)

    # the frequencies over which we calculate field quantities
    freqs = np.arange(minfreq, maxfreq, 10.0)
    # the central frequencies of each 10 MHz sub-band
    c_freqs = np.arange(minfreq + 5, maxfreq, 10.0)

    # get the trigger voltage
    Vtrig = trigger.trigger_level(c_freqs, prototype, antennas, trigger_sigma)

    # calculate the integrated noise voltage across the band
    Vn_spectrum = antenna.noise_voltage(c_freqs, prototype, antennas)

    # loop over each elevation angle
    for i in tqdm(np.arange(elev.shape[0])):

        # compute the geometric area at the desired elevation angles
        Ag = geometry.geometric_area(
            altitude, maxview, elev[i], 0, N=N[i], ice=icethickness
        )

        # if we didn't get any passing events, just skip this trial
        if Ag.emergence.size == 0:
            continue

        # get the exit probability at these elevation angles
        # this is a masked array and will be masked
        # if no tau's exitted at these angles
        Pexit, Etau = tauexit(90.0 - np.degrees(Ag.emergence))

        # get a random set of decay lengths at these energies
        decay_length = tauola.sample_range(Etau)

        # we now need the decay probability
        Pdecay = decay.probability(decay_length, Ag.dbeacon)

        # calculate the view angle from the decay point
        view = geometry.decay_view(Ag.view, Ag.dbeacon, decay_length)

        # and the sample the energy of the tau's
        Eshower = tauola.sample_tau_energies(Etau, N=Pdecay.size)

        # get the zenith angle at the decay
        # decay_zenith = geometry.decay_zenith(Ag.emergence, decay_length)

        # and get the altitude at the decay point
        decay_altitude = geometry.decay_altitude(
            Ag.emergence, decay_length, icethickness
        )

        # get the zenith angle at the exit point
        exit_zenith = np.pi / 2.0 - Ag.emergence
            
        # compute the voltage at each of these off-axis angles and at each frequency
        V = voltage(
            np.degrees(view),
            np.degrees(exit_zenith),
            decay_altitude,
            decay_length,
            Ag.axis,
            Ag.trials,
            Ag.beacon,
            freqs,
            Eshower,
            altitude,
            gain,
            antennas,
        )

        # sample the random noise amplitude density and integrate it
        # the Rayleigh-distributed noise vector projected onto the
        # Hpol-axis ends up as a half-normal distributed noise amplitude
        Vn = np.sqrt(
            np.sum(
                np.abs(
                    np.random.normal(loc=0, scale=Vn_spectrum ** 2.0, size=V.T.shape).T,
                ),
                axis=0,
            )
        )

        # and check for a trigger
        Ptrig = (np.sum(V, axis=0) / Vn) > trigger_sigma

        # the probability that noise triggers the payload
        PNtrig = Vn > Vtrig

        # the number of trials that we used in this iteration
        ntrials = float(N[i])

        # and save the various effective area coefficients at these angles
        geometric[i] = (Ag.area * np.sum(Ag.dot)) / N[i]
        pexit[i] = np.mean(Pexit)
        pdecay[i] = np.mean(Pdecay)
        ptrigger[i] = np.mean(Ptrig)
        effective_area[i] = np.sum(Ag.area * Ag.dot * Pexit * Pdecay * Ptrig) / ntrials
        Anoise[i] = np.sum(Ag.area * Ag.dot * Pexit * Pdecay * PNtrig) / ntrials

    # construct a dictionary of the arguments
    args = {
        "Enu": Enu,
        "altitude": altitude,
        "prototype": prototype,
        "maxview": maxview,
        "icethickness": icethickness,
        "antennas": antennas,
        "gain": gain,
        "trigger_sigma": trigger_sigma,
        "minfreq": minfreq,
        "maxfreq": maxfreq,
    }
    # and now return the computed parameters
    return EffectiveArea(
        N,
        np.degrees(elev),
        effective_area,
        geometric,
        pexit,
        pdecay,
        ptrigger,
        Anoise,
        args,
    )


def from_file(filename: str) -> EffectiveArea:
    """
    Load an effective area result from a file.
    Parameters
    ----------
    filename: str
        The filename containing a pickled EffectiveArea
    Returns
    -------
    Aeff: EffectiveArea
        The loaded effective area.
    """

    with open(filename, "rb") as f:
        return pickle.load(f)


def from_files(filenames: List[str]) -> EffectiveArea:
    """
    Load and combine effective area result from multiple files
    Parameters
    ----------
    filename: str
        The filename containing a pickled EffectiveArea
    Returns
    -------
    Aeff: EffectiveArea
        The loaded effective area.
    """

    # if we don't get any files, report an error
    if len(filenames) == 0:
        raise ValueError("No filenames were given to `from_files`.")

    # load the first file
    Aeff = from_file(filenames[0])

    # and load the rest of the files
    for f in filenames[1:]:
        A = from_file(f)
        if A.args["altitude"] == 3.87553:
            continue
        Aeff += A

    # and return the combined effective area
    return Aeff


# this lets us load pickled files from older poinsseta versions.
AcceptanceResult = EffectiveArea
