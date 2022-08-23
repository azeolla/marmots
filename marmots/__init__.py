import enum
import os
import os.path as path

# our global version number
__version__ = "0.1.0"

# the parent marmots directory
parent = path.dirname(path.dirname(path.abspath(__file__)))

# we use `data` unless the user overrides with marmots_DATA_DIR
data_directory = os.getenv("MARMOTS_DATA_DIR") or path.join(parent, "data")

# the directory where we store test figures
figdir = path.join(parent, *("tests", "figures"))

# a shower type enum to represent hadronic or electromagnetic


@enum.unique
class ShowerType(enum.IntEnum):
    """
    An enum representing hadronic or electromagnetic showers.
    """

    Hadronic = 0
    Electromagnetic = 1
