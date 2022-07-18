import poinsseta.flightpath as flightpath


def try_prototype(version: int):
    """
    Test that we can load BEACON data and access all variables.
    """

    # load the flight
    flight = flightpath.load_prototype(version)

    # get the number of elements in time
    N = flight.realTime.size

    # check that all the required variables are there
    assert flight.realTime.size == N
    assert flight.altitude.size == N
    assert flight.latitude.size == N
    assert flight.longitude.size == N


def test_beacon1():
    """
    Test loading BEACON 2018 data.
    """
    try_prototype(2018)


def test_beacon2():
    """
    Test loading BEACON 2019 data.
    """
    try_prototype(2019)
