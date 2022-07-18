import pickle
from os.path import dirname, exists

import numpy as np


def files_missing() -> bool:
    """
    Return True if tau LUT's are missing.
    """

    from poinsseta import figdir, data_directory

    # the filename we look for
    fname = f"{data_directory}/tauexit/2.0km_ice_midCS_stdEL/LUT_3e+16_eV.npz"

    # the filename we try to load to siff the files exist
    if not exists(fname):
        return True

    if not exists(f"{dirname(__file__)}/data" "/acceptance.pckle"):
        return True

    return False  # files are available!


def test_effective_area():
    """
    Produce a comparison of Claire's and my effective area.
    """

    # check if we have files
    if files_missing():
        return

    # otherwise we load our poinsseta packages
    from poinsseta.effective_area import AcceptanceResult
    from poinsseta import figdir

    # load the acceptance we previously calculated
    Aeff = pickle.load(open((f"{dirname(__file__)}/data" "/acceptance.pckle"), "rb"))

    # and create the plot of this Aeff result
    fig, ax = Aeff.plot()

    # load Claire's finail Aeff value
    claire_trig = np.loadtxt(
        f"{dirname(__file__)}/claire/claire_ptrigger_1e18.csv", delimiter=",",
    )

    # add ptrigger to the plot
    ax.semilogy(claire_trig[:, 0], claire_trig[:, 1], linestyle="--", color="grey")

    # and save the plot
    fig.savefig(f"{figdir}/effective_area.pdf")
