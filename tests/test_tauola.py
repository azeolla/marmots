import numpy as np

import poinsseta.tauola as tauola


def test_constants():
    """
    Perform some sanity checks on the tauola.py constants.
    """

    # make sure the shower counts all make sense
    assert tauola.Nshowers < tauola.decays.size
    assert tauola.Nem < tauola.Nshowers
    assert tauola.Nhad < tauola.Nshowers

    # check that the energies have been correctly sorted
    assert np.all(np.diff(tauola.Eem) >= 0)
    assert np.all(np.diff(tauola.Ehad) >= 0)


def test_sample_energy_fraction():
    """
    Perform some basic sanity checks on sample_energy_fraction
    """

    # generate a bunch of shower types
    stypes = tauola.sample_shower_type(1000)

    # and a bunch of energy fractions
    fractions = tauola.sample_energy_fraction(stypes)

    # and some sanity checks
    assert np.all(fractions > 0.0)
    assert np.all(fractions < 1.0)


def test_sample_shower_type():
    """
    Perform some sanity checks on sample_shower_type.
    """

    # sample a bunch of shower types
    stypes = tauola.sample_shower_type(1000)

    # and make sure we ony have 0's and 1's
    assert np.setdiff1d(stypes, np.asarray([0, 1])).size == 0


def test_sample_range():
    """
    Perform some sanity checks on sample_range.
    """

    # generate a bunch of energies in log-space
    Etau = np.logspace(17.0, 21.0, 1000)

    # sample a bunch of rangs
    ranges = tauola.sample_range(Etau)

    # make sure they are all positive
    assert np.all(ranges > 0.0)

    # and are all within some sensible limit
    assert np.all(ranges < 4.7e6)
