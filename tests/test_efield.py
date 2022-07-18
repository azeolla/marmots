"""
This module provides some tests of the field parameterization.
"""
import os.path as path

import matplotlib.pyplot as plt
import numpy as np

from poinsseta import data_directory, figdir


def files_missing() -> bool:
    """
    Return True if tau LUT's are missing.
    """

    # the filename we look for
    file_dir = path.join(data_directory, "beacon")
    fname = f"{file_dir}/interpolator_efields_37.0km.npz"

    # the filename we try to load to siff the files exist
    if not path.exists(fname):
        return True

    return False  # files are available!


def test_efield_plot():
    """
    Replicate the electric field plot from the ANITA diffuse paper.
    """
    if files_missing():
        return

    # the decay altitudes that we loop over
    decays = np.arange(10, dtype=np.float)

    # the view angles that we plot at
    view = np.linspace(0, 2.0, 1000)

    # pick a zenith angle
    zenith = 60.0  # degrees

    # load the electric field parameterization
    file_dir = path.join(data_directory, "beacon")

    interp_file = np.load(
        path.join(file_dir, "interpolator_efields_37.0km.npz"),
        allow_pickle=True,
        encoding="bytes",
    )
    efield_interpolator_list = interp_file["efield_interpolator_list"][()]

    # create the figure
    fig, ax = plt.subplots()

    # get the colors we use
    colors = plt.cm.inferno(np.linspace(0, 0.92, len(decays)))

    freqs = np.arange(180, 1200, 10)

    # loop over each decay altitude and plot
    for decay, color in zip(decays, colors):
        efields = []
        for freq in freqs:
            i_f_Lo = int(round(freq / 10 - 1))
            efields.append(efield_interpolator_list[i_f_Lo](zenith, decay, view))
        # evaluate the field and plot it
        ax.plot(
            view,
            1e3 * np.sum(efields, axis=0),
            color=color,
            label=f"Decay Altitude: {decay:0.0f} km",
        )

    # add some axis
    ax.set(xlabel=r"View Angle [$^\circ$]", ylabel=r"E$_{peak}$ [mV/m]")

    # and set the lower y limit as zero
    ax.set_ylim(bottom=0.0)
    ax.set_xlim([0, 2.0])

    # and add the legend
    plt.legend()

    # and save the plot
    fig.savefig(f"{figdir}/efield.pdf")
