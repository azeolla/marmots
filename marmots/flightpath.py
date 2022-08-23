"""
This module provides methods to load the location and elevation
data of various BEACON prototypes.
"""
import collections
import os.path as path
from typing import Any

import uproot3 as uproot

from marmots import data_directory


def load_prototype(prototype: int) -> Any:
    """
    Load the location and elevation data for a given prototype
     of BEACON (currently 2018/2019).

    The returned array has *at least* the following fields:

        realTime:  the time of each entry in unix time.
        altitude:  the payload altitude in m.
        latitude:  the payload latitude in degrees.
        longitude: the payload longitude in degrees.
        heading:   the payload heading in degrees.

    Parameters
    ----------
    prototype: int
        The prototype number to load.

    Returns
    -------
    flightpath: uproot.tree.Arrays
        A namedtuple-like class containing numpy arrays for each quantity.
    """

    # check for a valid version
    if prototype not in [2018, 2019]:
        raise ValueError(
            f"We currently only support BEACON{2018, 2019} (got: {prototype})"
        )

    # construct the filename for this prototype
    filename = path.join(data_directory, *("flightpaths", f"beacon{prototype}.root"))

    # open the ROOT file
    f = uproot.open(filename)

    # and load the ttree and return it to the user
    return f["adu5PatTree"].arrays(outputtype=collections.namedtuple)
