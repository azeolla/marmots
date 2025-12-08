"""
This module tests the TauExitLUT. Since these files are large,
we don't want to run these tests on Github/Travis etc.
We don't run them if the files don't exist.
"""
from os.path import dirname, exists

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import poinsseta.geometry as geometry
from poinsseta import data_directory, figdir
from poinsseta.tauexit import TauExitLUT


def files_missing() -> bool:
    """
    Return True if tau LUT's are missing.
    """

    # the filename we look for
    fname = f"{data_directory}/tauexit/2.0km_ice_midCS_stdEL/LUT_3e+16_eV.npz"

    # the filename we try to load to siff the files exist
    if not exists(fname):
        return True

    return False  # files are available!


def test_tau_exit_lut_filename():
    """
    Check if I can create a tau exit LUT by filename
    """

    # check if the files are missing
    if files_missing():
        return

    # for each ice thickness
    for thickness in [0.0, 1.0, 2.0, 3.0, 4.0]:

        # and for each energy exponent
        for energy in [15, 16, 17, 18, 19, 20, 21]:

            # and for each prefactor
            for mult in [1, 3]:

                # skip this combination
                if mult == 3 and energy == 21:
                    continue

                # make the filename
                filename = (
                    f"{thickness:.1f}km_ice_midCS_stdEL/LUT_{mult}e" f"+{energy}_eV"
                )

                # and try and open the LUT
                _ = TauExitLUT(filename=filename)


def test_tau_exit_lut():
    """
    Check if I can create a tau exit LUT by parameters.
    """

    # check if the files are missing
    if files_missing():
        return

    # for each ice thickness
    for thickness in [0.0, 1.0, 2.0, 3.0, 4.0]:

        # and for each energy exponent
        for energy in [1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21]:

            # and try and open the LUT
            _ = TauExitLUT(energy, thickness)


def test_tau_exit_call():
    """
    Check that I can successfully get single and array-like
    sampled energies and exit probabilities.
    """

    # check if the files are missing
    if files_missing():
        return

    # load a LUT
    t = TauExitLUT(energy=3e16, thickness=2.0)

    # get a single exit probability
    assert t.get_pexit(20.5) >= 0.0

    # and check that I can get an array of probabilities
    assert t.get_pexit(np.linspace(20, 30, 10)).size == 10

    # randomly sample an exit probability and energy
    Pexit, energy = t(1.0)
    assert Pexit >= 0

    # and do the same thing for an array
    Pexit, energy = t(np.linspace(80, 90, 10))
    assert Pexit.size == 10


def test_plot_pexit():
    """
    Produce a plot of the exit probability as a function
    of energy and exit angle.
    """

    # check if the files are missing
    if files_missing():
        return

    # create the figure
    fig, ax = plt.subplots(figsize=(4, 4))

    # the emergence angles at which we sample
    theta = np.linspace(90, 0, 180)

    # the thickness of the ice that we simulate
    thickness = 2.0

    # the energies we plot
    energies = [1e16, 1e17, 1e18, 1e19, 1e20]

    # and the colors
    colors = plt.cm.inferno(np.linspace(0, 1, len(energies) + 1))

    # loop over several energies
    for energy, color in zip(energies, colors):

        # and load the LUT
        lut = TauExitLUT(energy=energy, thickness=thickness)

        # evaluate the exit probability
        pexit = lut.get_pexit(90 - theta)

        # add the plot
        ax.semilogy(
            theta,
            pexit,
            label=(r"$\log_{10}E =$" f" {np.log10(energy):.0f}"),
            color=color,
        )

    # and some labels
    ax.set(xlabel=r"Emergence Angle ($^\circ$)", ylabel=r"Exit Probability")

    # and some limits
    ax.set_xlim([-1, 90])
    ax.set_ylim(bottom=1e-8)

    # and enable a logarithmic grid
    ax.grid(True, which="both")

    # and a title
    ax.set_title(r"$\tau$ Exit Probability")

    # and turn on the legend
    plt.legend()

    # and save the figure into the figures directory
    fig.savefig(f"{figdir}/exit_probability.pdf")


def test_plot_exit_energies():
    """
    Plot the distribution of tau exit energies for a 10^18 eV neutrino.
    """

    # the thickness of the ice that we simulate
    thickness = 2.0

    # and the energy that we simulate
    energy = 1e18

    # the number of tau's that we try and sample
    ntrials = 30_000

    # the number of steps in elevation angle
    nsteps = 68

    # assume the same parameters as in Claire's slides
    h = 37.5  # km

    # check if the files are missing
    if files_missing():
        return

    # create the figure
    fig, ax = plt.subplots(figsize=(4, 4))

    # the payload elevation angles
    elev = np.linspace(-40.0, -6.0, nsteps)

    # and calculate the emergence angles at which we sample
    thetas = np.degrees(geometry.emergence_angle(h, np.radians(elev)))

    # and the colors
    colors = plt.cm.inferno(np.linspace(0, 1, 3))

    # this array stores the median tau exit energy
    median = np.zeros_like(thetas)

    # and this stores the stdev of tau exit energies
    stdev = np.zeros_like(thetas)

    # and this stores the max and min tau exit energies
    max_energy = np.zeros_like(thetas)
    min_energy = np.zeros_like(thetas)

    # and the mask that we use to hide invalid values
    valid = np.zeros(thetas.size, dtype=bool)

    # and load the LUT
    lut = TauExitLUT(energy=energy, thickness=thickness)

    # we loop over every exit angle
    for i, theta in enumerate(thetas):

        # get an array of tau exit energies
        _, energies = lut((90 - theta) * np.ones(ntrials))

        # check if we got at least one tau
        valid[i] = energies.count() > 0

        # if we got one tau
        if valid[i]:
            # save the appropriate values
            median[i] = np.median(energies)
            stdev[i] = np.std(energies)
            max_energy[i] = np.max(energies)
            min_energy[i] = np.min(energies)

    # and now mask everything by the valid mask
    median = ma.array(median, mask=~valid)
    stdev = ma.array(stdev, mask=~valid)
    min_energy = ma.array(min_energy, mask=~valid)
    max_energy = ma.array(max_energy, mask=~valid)

    # and plot the min and max
    ax.fill_between(elev, min_energy, max_energy, color="lightgray", alpha=0.5)

    # we load Claire's results
    claire = np.loadtxt(
        f"{dirname(__file__)}/claire/median_tau_exit_energy_1e18.csv", delimiter=",",
    )

    # and plot it
    ax.semilogy(claire[:, 0], claire[:, 1], lw=1.0, color="blue", label="Claire")

    # plot the median tau energy
    ax.semilogy(elev, median, lw=1.0, color=colors[1], label="Tapioca")

    # and some labels
    ax.set(xlabel=r"Payload Elevation Angle [$^\circ$]", ylabel=r"Tau Energy [eV]")

    # and some limits
    ax.set_xlim([-40.0, 0])
    ax.set_ylim([1e14, energy])

    # and a title
    ax.set_title(r"$E_{\nu_{\tau}} = 10^{18}$ eV")

    # and enable a logarithmic grid
    ax.grid(True, which="both")

    # and create a legend
    plt.legend()

    # and save the figure into the figures directory
    fig.savefig(f"{figdir}/exit_energies.pdf")
