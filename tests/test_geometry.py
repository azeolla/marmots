import matplotlib.pyplot as plt
import numpy as np

from poinsseta import figdir
from poinsseta.constants import Re
from poinsseta.geometry import (
    geocentric_angle,
    geometric_area,
    horizon_angle,
    random_surface_angle,
    spherical_cap_area,
    spherical_segment_area,
    spherical_to_cartesian,
)


def test_plot_geometric_area():
    """
    Perform a basic plot of geometric area.
    """

    # the elevation angles and phi that we use
    elev = np.radians(np.linspace(-40, 0, 160))
    # elev = np.radians(np.linspace(-10, 0, 80))
    phi = 0.0

    # and we use a 38. km height
    height = 35.5

    # and the number of trials
    N = 500_000

    # and the maxview angle
    maxview = np.radians(1.0)

    # get the geometric area
    A_0 = np.asarray(
        [geometric_area(height, maxview, e, phi, N=N, ice=0.0)[0] for e in elev]
    )
    A_1 = np.asarray(
        [geometric_area(height, maxview, e, phi, N=N, ice=1.0)[0] for e in elev]
    )
    A_2 = np.asarray(
        [geometric_area(height, maxview, e, phi, N=N, ice=2.0)[0] for e in elev]
    )
    A_3 = np.asarray(
        [geometric_area(height, maxview, e, phi, N=N, ice=3.0)[0] for e in elev]
    )
    A_4 = np.asarray(
        [geometric_area(height, maxview, e, phi, N=N, ice=4.0)[0] for e in elev]
    )

    # and the colors we use for our various models
    colors = plt.cm.inferno(np.linspace(0, 1, 6))

    # create the figure
    fig, ax = plt.subplots(figsize=(4, 4))

    # let's load Claire's data - digitized from her final slides
    claire = np.loadtxt(f"{figdir}/../claire/claire_geometric_1e18.csv", delimiter=",")

    # and plot Claire's data
    ax.semilogy(claire[:, 0], claire[:, 1], lw=0.75, color="grey", label="Claire")

    # plot it
    ax.semilogy(
        np.degrees(elev),
        A_0,
        lw=0.75,
        color=colors[0],
        label=r"$D_{\mathrm{ice}} =0$ km",
    )
    ax.semilogy(
        np.degrees(elev),
        A_1,
        lw=0.75,
        color=colors[1],
        label=r"$D_{\mathrm{ice}} =1$ km",
    )
    ax.semilogy(
        np.degrees(elev),
        A_2,
        lw=0.75,
        color=colors[2],
        label=r"$D_{\mathrm{ice}} =2$ km",
    )
    ax.semilogy(
        np.degrees(elev),
        A_3,
        lw=0.75,
        color=colors[3],
        label=r"$D_{\mathrm{ice}} =3$ km",
    )
    ax.semilogy(
        np.degrees(elev),
        A_4,
        lw=0.75,
        color=colors[4],
        label=r"$D_{\mathrm{ice}} =4$ km",
    )

    # and enable a logarithmic grid
    ax.grid(True, which="both")

    # and some axis labels
    ax.set_xlabel(r"Payload Elevation Angle [deg]")
    ax.set_ylabel(r"$\langle A_g \rangle$ [km]")

    # and some limits
    ax.set_xlim([-40, 0])
    ax.set_ylim([1e-2, 2.2e2])

    # and add a legend
    plt.legend()

    # save the figure
    fig.savefig(f"{figdir}/geometric_area.pdf")


def test_spherical_segment_area():
    """
    Some basic tests of `spherical_segment_area`.
    """

    # compute the area of a reference spherical cap
    cap_area = spherical_cap_area(np.radians(32.0))

    # a spherical segment around the pole should be
    # the same area as a spherical cap
    np.testing.assert_allclose(spherical_segment_area(0, np.radians(32)), cap_area)

    # this should have half the cap area
    np.testing.assert_allclose(
        spherical_segment_area(0, np.radians(32), 0, np.pi), cap_area / 2.0
    )

    # make sure we can do the vectorized form
    np.testing.assert_allclose(
        spherical_segment_area(0, np.radians(32.0) * np.ones(10), 0, np.pi / 2.0),
        cap_area / 4.0,
    )

    # and this should just be the surface area of a sphere
    np.testing.assert_allclose(spherical_segment_area(0, np.pi), 4 * np.pi * Re * Re)


def test_spherical_cap_area():
    """
    Some basic tests of `spherical_cap_area`.
    """

    # a spherical cap of theta=0 has zero area
    np.testing.assert_array_equal(spherical_cap_area(0.0), 0.0)

    # and a cap with theta=pi is a full-sphere
    np.testing.assert_allclose(spherical_cap_area(np.pi), 4 * np.pi * Re * Re)

    # and a cap with theta=pi/2. is a half-sphere
    np.testing.assert_allclose(spherical_cap_area(np.pi / 2.0), 2 * np.pi * Re * Re)


def test_geocentric_angle():
    """
    Some basic tests of `geocentric_angle`.
    """

    # draw a bunch of random heights
    height = np.random.uniform(37.0, 40.0, size=5)
    elev = np.radians(np.random.uniform(-1.0, -3.0, size=5))

    # the Earth angle for an elevation angle of -90 should be zero
    np.testing.assert_allclose(
        geocentric_angle(height[0], -np.pi / 2.0), 0.0, atol=1e-14
    )

    # get abunch of elevation angles
    elev = np.radians(np.linspace(-90.0, -10.0, 20))

    # make sure that the Earth angle is a linearly increasing function of elev
    np.testing.assert_array_less(0, np.diff(geocentric_angle(37.0, elev)))


def test_horizon_angle():
    """
    Some basic tests of `horizon_angle`.
    """

    # makes sure we can call the non-vectorized vesion
    horizon = horizon_angle(37.0)
    assert horizon < np.radians(-6.0)

    # get a bunch of horizon angles
    horizons = horizon_angle(np.random.uniform(37.0, 40.0, size=1000))

    # and make sure they are within the ranges I calculated by hand.
    np.testing.assert_array_less(np.degrees(horizons), -6.1)
    np.testing.assert_array_less(-6.4, np.degrees(horizons))


def test_spherical_to_cartesian():
    """
    Some basic tests of `spherical_to_cartesian`.
    """
    # try converting the singular axes
    np.testing.assert_allclose(
        np.array([0, 0, 1.0]), spherical_to_cartesian(0, 0)[0], atol=1e-15
    )
    np.testing.assert_allclose(
        np.array([1.0, 0, 0.0]), spherical_to_cartesian(np.pi / 2.0, 0)[0], atol=1e-15
    )
    np.testing.assert_allclose(
        np.array([0, 1.0, 0.0]),
        spherical_to_cartesian(np.pi / 2.0, np.pi / 2.0)[0],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.array([0, 0, -1.0]), spherical_to_cartesian(np.pi, 0)[0], atol=1e-15
    )

    # check that the array form works with radius
    spherical = spherical_to_cartesian(
        np.pi / 2.0 * np.ones(10), np.pi / 2.0 * np.ones(10), r=1234
    )

    # check that the points match
    for i in np.arange(10):
        np.testing.assert_allclose(
            spherical[i, :], np.array([0, 1234.0, 0]), atol=1e-12
        )

    # and try some random theta values
    theta = np.random.uniform(0, np.pi, size=20)

    # convert them to cartesian
    cartesian = spherical_to_cartesian(theta, 0)

    # compute the angle between this and the z-axis
    thetam = np.arccos(np.einsum("ij,ij->i", cartesian, np.array([[0, 0, 1.0]])))

    # and make sure it matches
    np.testing.assert_allclose(theta, thetam)


def test_random_surface_angle():
    """
    Some basic tests of `random_surface_angle`.
    """

    # check that the scalar version works
    theta, phi = random_surface_angle(np.radians(10.0), np.radians(50.0))
    assert theta < np.radians(50.0)
    assert theta > np.radians(10.0)
    assert phi < 2 * np.pi
    assert phi > 0.0

    # and check that I can ask for multiple points
    theta, phi = random_surface_angle(np.radians(10.0), np.radians(50.0), N=10)
    assert theta.size == 10
    assert phi.size == 10

    # the number of random points we use for the test
    N = 100

    # draw a bunch of random theta min's and theta max's
    theta_min = np.random.uniform(0, np.radians(20.0), size=N)
    theta_max = np.random.uniform(np.radians(30.0), np.radians(80.0), size=N)

    # draw a bunch of random phi min's and phi max's
    phi_min = np.random.uniform(0, np.pi / 2.0, size=N)
    phi_max = np.random.uniform(np.pi / 2.0, np.pi, size=N)

    # draw a bunch of random surface angles
    theta, phi = random_surface_angle(theta_min, theta_max, phi_min, phi_max, N=N)

    # check that the right number of points was sampled
    assert theta.size == N

    # check the range of every theta
    np.testing.assert_array_less(theta, theta_max)
    np.testing.assert_array_less(theta_min, theta)

    # and the range of every phi
    np.testing.assert_array_less(phi, phi_max)
    np.testing.assert_array_less(phi_min, phi)
