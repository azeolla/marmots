"""
This module provides free-functions for calculating various
needed geometric quantities and formulas.
"""
from typing import Any, NamedTuple, Tuple

import numpy as np

from poinsseta.constants import Re

__all__ = [
    "view_angle",
    "spherical_segment_area",
    "spherical_cap_area",
    "points_on_earth",
    "horizon_angle",
    "geocentric_angle",
    "random_surface_angle",
    "random_surface_point",
    "spherical_to_cartesian",
    "geometric_area",
]

# create a named tuple to store our geometry information
GeometricArea = NamedTuple(
    "GeometricArea",
    [
        ("area", np.ndarray),
        ("dot", np.ndarray),
        ("emergence", np.ndarray),
        ("view", np.ndarray),
        ("dbeacon", np.ndarray),
        ("beacon", np.ndarray),
        ("trials", np.ndarray),
        ("axis", np.ndarray),
        ("axes", np.ndarray)
    ],
)


def geometric_area(
    height: float,
    maxview: float,
    elev: float,
    phi: float,
    N: int = 10_000,
    ice: float = 0.0,
) -> GeometricArea:
    """
    Compute the geometric area on the surface of the Earth "illuminated"
    within a maximum `view` angle around a source elevation `elev` and
    a source `phi` from an an altitude `height`.

    Parameters
    ----------
    height: float
        The observation heights (km).
    maxview: float
        The maximum view angle from the payload (radians).
    elev: float
        The elevation angle (-nve) of the source at the payload (radians).
    phi: float
        The azimuthal angle of the source (radians).
    N: int
        The number of trials to evaluate the integral.
    ice: float
        The constant thickness of the ice (km)

    Returns
    -------
    area: float
        The geometric area at each elevation angle (km^2).
    emergence: np.ndarray
        The detected emergence angles for each trial (radians)
    view: np.ndarray
        The detected view angles for each trial (view)
    """

    # get the vector from the center of the area to the payload
    axis = spherical_to_cartesian(np.pi / 2.0 + elev, phi + np.pi, r=1.0)

    # generate N random points in the corresponding segment
    trials, A0 = points_on_earth(height, elev, phi, maxview, N, ice=ice)

    # compute the view angle of each of these points
    view = view_angle(height, trials, axis)

    # find all the points that pass the view angle cut
    trials = trials[view < maxview, :]

    # compute the dot product of each trial point with the axis vector
    dot = np.einsum(
        "ij,ij->i", axis, trials / np.linalg.norm(trials, axis=1).reshape((-1, 1))
    )

    # and mask those trials whose dot product < 0 - since trials are on
    # the surface of the Earth, this would require the RF to propagate
    # through the Earth which is going to render the event undetectable.
    valid = dot > 0

    # mask the dot products and view angle
    dot = dot[valid]
    view = view[view < maxview][valid]

    # compute the average emergence angle for this elevation
    emergence = np.pi / 2.0 - np.arccos(dot) if dot.size else np.asarray([])

    # save the location of BEACON
    beacon = np.asarray([0, 0, Re + height])

    # the vector from the exit point to BEACON
    to_beacon = (beacon - trials)[valid]

    # get the distance from the exit point to BEACON
    dbeacon = np.linalg.norm(to_beacon, axis=1)

    # and compute the normalized direction vector to BEACON
    axes = to_beacon / dbeacon.reshape((-1, 1))

    # and we are done
    return GeometricArea(A0, dot, emergence, view, dbeacon, beacon, trials[valid], axis, axes)


def decay_view(
    exitview: np.ndarray, dbeacon: np.ndarray, trange: np.ndarray
) -> np.ndarray:
    """
    Given the view angle at the exit location, the distance to BEACON
    along this view angle, and the tau lepton range, calculate the
    view angle (radians) from the decay point.

    This formula can easily be derived by applying the sine rule
    and the cosine rule to the triangle formed by `trange`, `drange`,
    and `exitview`.

    Parameters
    ----------
    exitview: np.ndarray
        The view angle (radians) at the exit location.
    dbeacon: np.ndarray
        The distance from the exit point to BEACON along the view angle.
    trange: np.ndarray
        The distance the tau travels before decaying.

    Returns
    -------
    decayview: np.ndarray
        The view angle of BEACON from the location of the tau decay.
    """

    # this is sin^2(phi) where phi is our view angle from decay
    s2phi = (dbeacon * np.sin(exitview)) ** 2.0 / (
        trange * trange + dbeacon * dbeacon - 2 * trange * dbeacon * np.cos(exitview)
    )

    # and we are done
    return np.arcsin(np.sqrt(s2phi))


def view_angle(height: np.ndarray, point: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Given the height of the observer (in km), the location of the observation
    point `point` in geocentric coordinates, and the particle `axis`, calculate
    the angle of the observer from the particle's frame i.e. the view angle.

    Parameters
    ----------
    height: np.ndarray
       A (N, 1)-length array containing detector heights (in km).
    point: np.ndarray
       A (N, 3)-length array containing obspoint locations in
       geocentric (km) coordinates.
    axis: np.ndarray
       A (N, 3)-length array containing the normalized
       axis of the particle velocity.

    Returns
    -------
    view: np.ndarray
       The view angle for each decay point (in radians).

    """
    # make sure points is at-least 2D
    point = np.atleast_2d(point)
    axis = np.atleast_2d(axis)

    # we currently assume that the detector is at (0, 0, Re+h)
    obspoint = np.zeros((point.shape[0], 3))
    obspoint[:, 2] = height + Re

    # calculate the vector from the point to the obs. points
    view = obspoint - point

    # and make sure that view is normalized
    view /= np.linalg.norm(view, axis=1).reshape((-1, 1))

    # make sure that axis is normalized (just in case)
    # NB: don't use /= as it will overwrite the argument
    axis = axis / np.linalg.norm(axis, axis=1).reshape((-1, 1))

    # and compute the view angle and we are done
    return np.arccos(np.einsum("ij,ij->i", view, axis))


def points_on_earth(
    height: float,
    elev: float,
    phi: float,
    view: float,
    N: int = 1_000,
    ice: float = 0.0,
) -> np.ndarray:
    """
    Sample `N` 3D vectors from the patch on the Earth centered at a given
    pyaload and elevation angle with a maximum view angle of `maxview`.

    Parameters
    ----------
    height: float
        The height of the observation point in km.
    elev: float
        The payload elevation angle in radians (-ve below the horizon)
    phi: float
        The payload azimuth angle in radians.
    view: float
        The maximum view angle in radians.
    N: int
        The number of sample points to generater.
    ice: float
        The constant thickness of the ice (km).

    Returns
    -------
    points: np.ndarray
        A (N, 3) ndarray containing the generated random points.
    area: float
        The trapezoidal surface area sampled from (in km^2).
    """

    # get the min/maximum view angle
    minview = elev + view
    maxview = elev - view

    # get the geocentric angle formed by this view angle
    theta_min = geocentric_angle(height, minview, ice=ice)
    theta_max = geocentric_angle(height, maxview, ice=ice)

    # and the corresponding phi ranges
    phi_min = np.atleast_1d(phi - maxview)
    phi_max = np.atleast_1d(phi + maxview)

    # if the geocentric angle is smaller than the maxview angle
    # phi can wrap in weird ways. So if we are considering the tiny
    # patch near the pole (right below the payload), we use a spherical cap.
    phi_min[theta_max < maxview] = 0
    phi_max[theta_max < maxview] = 2 * np.pi

    # generate the random trial points
    trials = random_surface_point(
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
        N=N,
        ice=ice,
    )

    # and get the total geometric area
    A0 = spherical_segment_area(
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
        ice=ice,
    )

    # and return both
    return trials, A0


def spherical_segment_area(
    theta_min: np.ndarray,
    theta_max: np.ndarray,
    phi_min: np.ndarray = 0.0,
    phi_max: np.ndarray = 2 * np.pi,
    ice: float = 0.0,
) -> np.ndarray:
    """
    Calculate the surface area (in km^2) of a spherical segment between
    polar angles `theta_min` and `theta_max` and between azimuthal angles
    `phi_min` and `phi_max`.

    See: https://en.wikipedia.org/wiki/Spherical_segment

    Parameters
    ----------
    theta_min: np.ndarray
       The minimum polar angle of the segment (radians).
    theta_max: np.ndarray
       The maximum polar angle of the segment (radians).
    phi_min: np.ndarray
       The minimum azimuthal angle of the segment (radians).
    phi_max: np.ndarray
       The maximum azimuthal angle of the segment (radians).
    ice: float
       The constant thickness of the ice [km].

    Returns
    -------
    area: np.ndarray
       The area of the spherical segment (in km^2).
    """
    return np.abs(
        ((Re + ice) ** 2.0)
        * (phi_max - phi_min)
        * np.abs(np.cos(theta_min) - np.cos(theta_max))
    )


def spherical_cap_area(theta: np.ndarray, ice: float = 0.0) -> np.ndarray:
    """
    Calculate the surface area (in km^2) of a spherical cap with polar
    angle theta (in radians).

    Parameters
    ----------
    theta: np.ndarray
        The polar angle of each spherical cap (in radians).
    ice: float
        The constant thickness of the ice [km].

    Returns
    -------
    area: np.ndarray
        The area of the spherical cap (in km^2)
    """
    return 2 * np.pi * ((Re + ice) ** 2.0) * (1 - np.cos(theta))


def horizon_angle(height: np.ndarray, ice: float = 0.0) -> np.ndarray:
    """
    Calculate the horizon angle (in radians)
    for a given set of heights (in km).

    Parameters
    ----------
    height: np.ndarray
       The height of the viewing point (km).
    ice: float
       The constant ice thickness (km)

    Returns
    -------
    horizon: np.ndarray
       The horizon angle (in radians).
    """
    return -np.arccos((Re + ice) / (Re + height))


def geocentric_angle(
    height: np.ndarray, elev: np.ndarray, ice: float = 0.0, reflect: bool = True,
) -> np.ndarray:
    """Given a viewing height (in km) and a set of payload elevation angles (in
    radians), calculate the geocentric Earth angle of the intersect of a vector
    along the elevation angle vector with a spherical Earth.

    If the elevation angle vector does not intersect the Earth, and `reflect` is
    True, reflect around the horizon angle so that we can smoothly calculate the
    area of a spherical patch over the horizon.

    See:
        https://math.stackexchange.com/questions/209271/
        given-an-altitude-and-a-viewing-angle-how-do-i-determine-the-distance-of-the-vie

    for a diagram and derivation of the formula used in this implementation.

    Parameters
    ----------
    height: np.ndarray
       The height (in km) of each viewing position.
    elev: np.ndarray
       The payload elevation angle (in radians).
    ice: float
       The thickness of the ice [km].
    reflect: bool
       If True, reflect above horizon angles over the horizon.
    Returns
    -------
    earth_angle: np.ndarray
       The geocentric Earth angle (in radians)

    """

    # make sure that elev, height are atleast 1D
    elev = np.atleast_1d(elev)
    height = np.atleast_1d(height)

    # calculate the horizon angle at each of these locations
    horizon = horizon_angle(height, ice)

    # the indices that we need to reflect around horizon
    above_horizon = elev > horizon

    # reflect the values around the horizon
    elev[above_horizon] = 2 * horizon - elev[above_horizon]

    # compute the 'theta' in the reference
    theta = np.pi / 2.0 + elev

    # compute sin(phi)
    sphi = ((Re + height) / (Re + ice)) * np.sin(theta)

    # create an array to store the result
    earth_angle: np.ndarray = np.zeros_like(sphi)

    # by default, we use the horizon angle
    # earth_angle[:] = -horizon_angle(height, ice)

    # if sphi < 1., arcsin is defined and we intersect the Earth
    # intersect = np.abs(sphi) <= 1.0

    # and for points that intersect, we overwrite with the true earth angle
    # earth_angle[intersect] = np.arcsin(sphi[intersect]) - theta[intersect]

    # and compute the earth angle
    earth_angle[~above_horizon] = (
        np.arcsin(sphi[~above_horizon]) - theta[~above_horizon]
    )

    # and correct the angles that are "beyond" the horizon
    if reflect:
        earth_angle[above_horizon] = (
            np.pi - np.arcsin(sphi[above_horizon]) - theta[above_horizon]
        )
    else:
        earth_angle[above_horizon] = horizon

    # and we are done
    return earth_angle


def emergence_angle(
    height: np.ndarray, elev: np.ndarray, ice: float = 0.0
) -> np.ndarray:
    """
    Given a viewing height (in km) and a set of payload elevation angles (in
    radians), compute the emergence angle from the surface intersection of the
    elevation angle looking at the payload.

    See:
        https://math.stackexchange.com/questions/209271/
        given-an-altitude-and-a-viewing-angle-how-do-i-determine-the-distance-of-the-vie

    for a diagram and derivation of the formula used in this implementation.

    Parameters
    ----------
    height: np.ndarray
       The height (in km) of each viewing position.
    elev: np.ndarray
       The payload elevation angle (in radians).
    ice: float
       The thickness of the ice [km].

    Returns
    -------
    emergence_angle: np.ndarray
       The emergence angle (in radians)
    """

    # make sure that elev, height are atleast 1D
    elev = np.atleast_1d(elev)
    height = np.atleast_1d(height)

    # compute the 'theta' in the reference
    theta = np.pi / 2.0 + elev

    # compute sin(phi)
    sphi = ((Re + height) / (Re + ice)) * np.sin(theta)

    # create an array to store the result
    emerg_angle: np.ndarray = np.zeros_like(sphi)

    # if sphi < 1., arcsin is defined and we intersect the Earth
    intersect = np.abs(sphi) <= 1.0

    # and for points that intersect, we overwrite with the true earth angle
    emerg_angle[intersect] = np.pi / 2.0 - np.arcsin(sphi[intersect])

    # and we are done
    return emerg_angle


def elevation_angle(
    height: np.ndarray, emergence: np.ndarray, ice: float = 0.0
) -> np.ndarray:
    """
    Given a viewing height (in km) and a set of surface elevation angles (in
    radians), compute the payload elevation angle.

    See:
        https://math.stackexchange.com/questions/209271/
        given-an-altitude-and-a-viewing-angle-how-do-i-determine-the-distance-of-the-vie

    for a diagram and derivation of the formula used in this implementation.

    Parameters
    ----------
    height: np.ndarray
       The height (in km) of each viewing position.
    emergence: np.ndarray
       The emergence angles at the surface (radians).
    ice: float
       The thickness of the ice [km].

    Returns
    -------
    elevation_angle: np.ndarray
       The payload elevation angle (radians).
    """

    # make sure that elev, height are atleast 1D
    emergence = np.atleast_1d(emergence)
    height = np.atleast_1d(height)

    # compute the sine of the elevation angle
    sine_elev: np.ndarray = ((Re + ice) / (Re + height)) * np.sin(np.pi - emergence)

    # and finally compute the elevation angle
    elev: np.ndarray = -np.arcsin(sine_elev)

    # and return the actual elevation angle
    return elev


def random_surface_point(
    theta_min: np.ndarray = 0,
    theta_max: np.ndarray = np.radians(22.5),
    phi_min: np.ndarray = 0,
    phi_max: np.ndarray = 2 * np.pi,
    ice: float = 0.0,
    **kwargs: Any,
) -> np.ndarray:
    """
    Return a random point on the surface of a spherical cap or segment
    in geocentric coordinates (in km).

    Parameters
    ----------
    theta_min: np.ndarray
       The minimum polar angle to be drawn (radians).
    theta_max: np.ndarray
        The maximum polar angle to be drawn (radians).
    phi_min: np.ndarray
       The minimum azimuthal angle to be drawn (radians).
    phi_max: np.ndarray
       The maximum azimuthal angle to be drawn (radians).
    ice: float
       The thickness of the ice [km].

    Returns
    -------
    points: np.ndarray
        A (N, 3) ndarray containing the geocentric coordinates (km).
    """

    # draw a pair of random angles (theta, phi) in spherical coordinates
    theta, phi = random_surface_angle(theta_min, theta_max, phi_min, phi_max, **kwargs)

    # and convert this spherical point into cartesian coordinates
    return spherical_to_cartesian(theta, phi, r=(Re + ice))


def random_surface_angle(
    theta_min: np.ndarray = 0,
    theta_max: np.ndarray = np.radians(22.5),
    phi_min: np.ndarray = 0,
    phi_max: np.ndarray = 2 * np.pi,
    N: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a random (theta, phi) point on the surface of the Earth drawn
    uniformly in solid angle. If no arguments are provided, this draws points
    from a spherical cap roughly the size of Antarctica.

    `theta_min` and `theta_max` must be identical length vectors or scalars.
    `phi_min` and `phi_max` must be identical length vectors or scalars.

    Parameters
    ----------
    theta_min: np.ndarray
       The minimum polar angle to be drawn (radians).
    theta_max: np.ndarray
        The maximum polar angle to be drawn (radians).
    phi_min: np.ndarray
       The minimum azimuthal angle to be drawn (radians).
    phi_max: np.ndarray
       The maximum azimuthal angle to be drawn (radians).
    N: int
       The number of points to generate if arguments are scalars.
    Returns
    -------
    theta: np.ndarray
       The randomly sampled polar angles.
    phi: np.ndarray
       The randomly sampled azimuthal angles.
    """

    # draw a random polar angle uniformly in solid angle
    theta = np.arccos(np.random.uniform(np.cos(theta_min), np.cos(theta_max), size=N))

    # and a uniform azimuthal angle
    phi = np.random.uniform(phi_min, phi_max, size=N)

    # and we are done
    return theta, phi


def spherical_to_cartesian(
    theta: np.ndarray, phi: np.ndarray, r: np.ndarray = 1.0
) -> np.ndarray:
    """
    Convert an array of (theta, phi, r) points to (N, 3) Cartesian vectors.

    Parameters
    ----------
    theta: np.ndarray
       The polar angle (in radians).
    phi: np.ndarray
       The azimuthal angle (in radians)
    r: np.ndarray
       The radius of the points (defaults to 1. for unit sphere)

    Returns
    -------
    points: np.ndarray
       A (N, 3) array containing the Cartesian coordinates.
    """

    # make sure they are both atleast 1D
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)

    # create the storage for the cartesian points
    cartesian = np.zeros((theta.size, 3))

    # and fill in the values
    cartesian[:, 0] = np.sin(theta) * np.cos(phi)
    cartesian[:, 1] = np.sin(theta) * np.sin(phi)
    cartesian[:, 2] = np.cos(theta)

    # and we are done
    return r * cartesian


def decay_zenith(emergence: np.ndarray, decay_length: np.ndarray) -> np.ndarray:
    """
    Give the emergence angle at the surface, calculate the zenith angle at
    the decay point given the decay_length.

    Parameters
    ----------
    emergence: np.ndarray
        The emergence angle at the surface (radians)
    decay_length: np.ndarray
        The decay length from the surface (in km).

    Returns
    -------
    zenith_angle: np.ndarray
        The trajectory zenith angle at the decay point.
    """

    # calculate the local zenith angle
    local_zenith = np.pi / 2.0 - emergence

    # use the law of cosines to calculate the length of
    # the radial vector at the decay point
    r2 = Re ** 2.0 + decay_length ** 2.0 + 2 * Re * decay_length * np.cos(local_zenith)

    # and use the sine-rule to convert this to a zenith at decay
    zenith: np.ndarray = np.arcsin(
        np.sqrt((Re ** 2.0 / r2) * (np.sin(local_zenith) ** 2.0))
    )

    # and we are done
    return zenith


def decay_altitude(
    emergence: np.ndarray, decay_length: np.ndarray, thickness: float
) -> np.ndarray:
    """
    Give the emergence angle at the surface, and the decay length,
    calculate the altitude of the decay point.


    Parameters
    ----------
    emergence: np.ndarray
        The emergence angle at the surface (radians)
    decay_length: np.ndarray
        The decay length from the surface (in km).
    thickness: float
        The ice thickness (in km).

    Returns
    -------
    altitude: np.ndarray
        The altitude of the decay point (km).
    """

    # calculate the local zenith angle
    local_zenith = np.pi / 2.0 - emergence

    # the radius at sea-level
    Rsea = Re + thickness

    # and use the cosine rule to calculate the geocentric distance
    geocentric = np.sqrt(
        Rsea ** 2.0
        + decay_length ** 2.0
        + 2 * Rsea * decay_length * np.cos(local_zenith)
    )

    # and convert that into an altitude ASL
    altitude: np.ndarray = geocentric - Re

    return altitude
