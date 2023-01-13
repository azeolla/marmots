"""
This module provides free-functions for calculating various
needed geometric quantities and formulas.
"""
from shutil import ReadError
from typing import Any, NamedTuple, Tuple

import numpy as np
import astropy.units as u
import astropy.coordinates as coordinates
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, split, transform
from math import remainder

from marmots.constants import Re
import pymap3d as pm

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
    "obs_zenith_azimuth",
    "distance_to_horizon"
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
        ("stations", np.ndarray),
        ("trials", np.ndarray),
        ("axis", np.ndarray),
        ("to_beacon", np.ndarray),
        ("orientations", np.ndarray),
        ("fov", np.ndarray),
    ],
)

def geometric_area(
    beacon: coordinates.EarthLocation,
    source: coordinates.SkyCoord,
    altaz: Any,
    maxview: float,
    orientations: np.ndarray,
    fov: np.ndarray,
    N: int = 10_000,
):
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

    theta = (90*u.deg - source.lat).to(u.rad)
    phi = source.lon.to(u.rad)

    # the particle axis
    axis = -spherical_to_cartesian(theta, phi, r=1.0)[0]

    # generate N random points in the corresponding segment
    trials, A0, stations, orientations, fov = points_on_earth(beacon, altaz, maxview, orientations, fov, N)

    if (trials.size == 0):

        return GeometricArea(A0, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), axis, np.array([]), np.array([]), np.array([]))

    else:

        # compute the view angle of each of these points
        view = view_angle(trials, stations, axis)

        # find all the points that pass the view angle cut
        trials = trials[view < maxview]

        # compute the dot product of each trial point with the axis vector
        dot = np.dot(trials/np.linalg.norm(trials, axis=1)[:,None], axis)

        # and mask those trials whose dot product < 0 - since trials are on
        # the surface of the Earth, this would require the RF to propagate
        # through the Earth which is going to render the event undetectable.
        valid = dot > 0

        # mask the dot products and view angle
        dot = dot[valid]
        view = view[view < maxview][valid]

        # compute the average emergence angle for this elevation
        emergence = np.pi/2.0*u.rad - np.arccos(dot) if dot.size else np.asarray([])

        # the vector from each exit point to each station
        to_beacon = (stations[:,None] - trials[valid])

        # get the distance from the exit point to BEACON
        dbeacon = np.linalg.norm(to_beacon, axis=2)

        # and compute the normalized direction vector to BEACON
        to_beacon = to_beacon / dbeacon[:,:,None]

        # and we are done
        return GeometricArea(A0, dot, emergence, view, dbeacon, stations, trials[valid], axis, to_beacon, orientations, fov)


def decay_view(
    decay_point: np.ndarray, axis: np.ndarray, station: np.ndarray,
) -> np.ndarray:

    d = station - decay_point
    
    return np.arccos(np.dot(d/np.linalg.norm(d, axis=1)[:,None], axis)) 


def view_angle(point: np.ndarray, obspoint: np.ndarray, axis: np.ndarray) -> np.ndarray:
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

    # calculate the vector from the point to the obs. points
    view = obspoint[:, None] - point 
    
    # and make sure that view is normalized
    view = view/np.linalg.norm(view, axis=2)[:,:,None]

    # make sure that axis is normalized (just in case)
    # NB: don't use /= as it will overwrite the argument
    axis = axis / np.linalg.norm(axis)

    # and compute the view angle and we are done
    return np.arccos(np.dot(view, axis)).to(u.deg)


def points_on_earth(
    beacon: coordinates.EarthLocation,
    altaz: Any,
    view: float,
    orientations: np.ndarray,
    fov: np.ndarray,
    N: int = 1_000,
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

    minview = altaz.alt.to(u.rad) + view
    maxview = altaz.alt.to(u.rad) - view

    # angle to the horizon
    horizon = horizon_angle(beacon.height.to(u.km), radius=Re)

    # if minview is above the horizon, it should be set to the horizon
    minview[minview > horizon] = horizon[minview > horizon]

    # if maxview is above the horizon, we don't view the Earth at all
    minview[maxview > horizon] = np.nan
    maxview[maxview > horizon] = np.nan

    minaz = altaz.az.to(u.rad) - view
    maxaz = altaz.az.to(u.rad) + view

    # if we're looking closing to straight down then maxview will be greater than 90 deg.
    
    maxview[maxview < -np.pi/2*u.rad] = -np.pi/2*u.rad

    # find the indices in which the Earth is not in sight
    invalid = (np.isnan(minview) | np.isnan(maxview))

    if (np.sum(invalid) == beacon.size):
        rtrials = np.array([])
        A0 = 0 
        stations = np.array([])
    else:
        # get the geocentric angle formed by these view angles
        theta_min = geocentric_angle(beacon.height[~invalid], maxview[~invalid], radius=Re)
        theta_max = geocentric_angle(beacon.height[~invalid], minview[~invalid], radius=Re)
        phi_min = -minaz[~invalid] # negative sign to account for azimuth being positive towards the East
        phi_max = -maxaz[~invalid]

        # generate the random trial points
        trials = random_surface_point(
                theta_min=theta_min,
                theta_max=theta_max,
                phi_min=phi_min,
                phi_max=phi_max,
                N=N,
                radius=Re,
            )

        # convert to cartesian (makes the rotation much easier)
        point1 = spherical_to_cartesian(theta_min, phi_min, r=Re)
        point2 = spherical_to_cartesian(theta_min, phi_max, r=Re)
        point3 = spherical_to_cartesian(theta_max, phi_max, r=Re)
        point4 = spherical_to_cartesian(theta_max, phi_min, r=Re)

        points = np.array(list(zip(point1 , point2, point3, point4)))

        station_theta = (90*u.deg - beacon.lat[~invalid]).to(u.rad)
        station_phi = beacon.lon[~invalid].to(u.rad)
        station_height = beacon.height[~invalid].to(u.km)

        # rotate the area vertices to their appropriate place on Earth
        rotated = np.array(list(map(rotate_earth, points, station_theta, station_phi)))

        # rotate the trial points to their appropriate place on Earth
        rtrials = np.array(list(map(rotate_earth, trials, station_theta, station_phi)))*u.km

        # convert the vertices back to spherical
        rthetaphi = np.array(list(map(cartesian_to_spherical, rotated*u.km)))*u.rad

        # find the vertices in which some have negative phi and some have postive phi (the shape passes over the prime or anti meridian)
        sum = (np.sum(rthetaphi[:,:,2] > 0*u.rad, axis=1))
        # the shapes wrap correctly over the prime meridian (phi = 0), but mess up at the antimeridian (phi = pi)
        wrapped_over_antimeridian = rthetaphi[(sum != 0) & (sum != 4) & (np.mean(abs(rthetaphi[:,:,2]), axis=1) > np.pi/2*u.rad)]
        negative_phi = wrapped_over_antimeridian[:,:,2] < 0
        wrapped_over_antimeridian[:,:,2][negative_phi] += 2*np.pi*u.rad # add 2pi to the negative phi values
        rthetaphi[(sum != 0) & (sum != 4) & (np.mean(abs(rthetaphi[:,:,2]), axis=1) > np.pi/2*u.rad)] = wrapped_over_antimeridian

        # perform a sinuisodal projection on the vertices 
        projection = np.swapaxes(np.array(list(map(project, rthetaphi[:,:,0], 90*u.deg - rthetaphi[:,:,1].to(u.deg), rthetaphi[:,:,2].to(u.deg)))), 1, 2)

        # and get the total geometric area
        A0 = union_area(projection)

        # the cartesian coordinates of the stations with land in view
        stations = spherical_to_cartesian(station_theta, station_phi, Re+station_height)
        orientations = orientations[~invalid]
        fov = fov[~invalid]

    # and return the trials, area, and valid stations
    return rtrials, A0, stations, orientations, fov


def rotate_earth(point, theta, phi):

    theta = -theta
    phi += np.pi*u.rad

    Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

    Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

    return np.einsum('ij, ...j->...i', Rz, np.einsum('ij, ...j->...i', Ry, point))


def cartesian_to_spherical(point):

    spherical = np.zeros((point.shape[0], 3))

    spherical[:,0] = np.linalg.norm(point, axis=1)*u.km
    spherical[:,1] = np.arccos(point[:,2]/np.linalg.norm(point, axis=1))*u.rad
    spherical[:,2] = np.arctan2(point[:,1], point[:,0])*u.rad

    return spherical


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


def project(radius, latitude, longitude):
    lat_dist = np.pi * radius / 180
    y = latitude * lat_dist 
    x = longitude * lat_dist * np.cos(latitude.to(u.rad))
    return x.value, y.value


def union_area(
    points: np.ndarray,
):
    # find the union of all the polygons
    polygon = unary_union([Polygon(x) for x in points.tolist()])

    # define the the boundary of the sinusoidal projection (the antimeridian)
    left_bound = project(Re, np.arange(-90, 91,1)*u.deg, -180*u.deg)
    right_bound = project(Re, np.arange(-90, 91,1)*u.deg, 180*u.deg)
    a = np.array([left_bound[0], left_bound[1]]).T
    b = np.flip(np.array([right_bound[0], right_bound[1]]).T, axis=0)
    ring = LineString(np.array([a,b]).reshape(2*181,2))

    # split the union-polygon along this boundary
    split_polygon = split(polygon, ring)

    # polygons that lie outside the boundary must be translated to the other side
    geom = []
    for i in range(split_polygon.geoms.__len__()):
        if (Polygon(ring).contains(split_polygon.geoms.__getitem__(i).buffer(-1e-3)) == False):
            geom.append(transform(translate, split_polygon.geoms.__getitem__(i)))
        else:
            geom.append(split_polygon.geoms.__getitem__(i))

    # now that everything is in the proper spot, take the union again
    final_polygon = unary_union(geom)
            
    return final_polygon.area


def translate(x,y):
    lat_dist = np.pi * Re.value / 180
    lat = (y/lat_dist)*u.deg
    lon = (x/(lat_dist * np.cos(lat.to(u.rad))))*u.deg
    new_lon = remainder(lon.value, 360)*u.deg
    new_x, new_y = project(Re, lat, new_lon)
    return tuple([new_x, new_y])


def horizon_angle(height: np.ndarray, radius: float = Re) -> np.ndarray:
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
    return -np.arccos((radius) / (radius + height))


def geocentric_angle(
    height: np.ndarray, elev: np.ndarray, radius: float = Re,
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

    horizon = horizon_angle(height, radius)

    earth_angle: np.ndarray = np.zeros_like(elev)

    earth_angle[elev == horizon] = -horizon[elev == horizon]

    # compute the 'theta' in the reference
    theta = (np.pi/2.0)*u.rad + elev

    # compute sin(phi)
    sphi = ((radius + height) / radius) * np.sin(theta)

    earth_angle[elev < horizon] = (np.arcsin(sphi[elev < horizon]) - theta[elev < horizon])

    earth_angle[elev > horizon] = np.nan
    
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
    N: int = 1,
    radius: np.ndarray = Re,
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
    theta = np.arccos(list(map(np.random.uniform, np.cos(theta_min.repeat(N).reshape(theta_min.size, N)), 
          np.cos(theta_max.repeat(N).reshape(theta_max.size, N)))))
    
    phi = np.array(list(map(np.random.uniform, phi_min.repeat(N).reshape(phi_min.size, N), 
         phi_max.repeat(N).reshape(phi_max.size, N))))

    # and convert this spherical point into cartesian coordinates
    return np.array(list(map(spherical_to_cartesian, theta, phi, radius.repeat(theta.size))))


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
    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)

    # create the storage for the cartesian points
    cartesian = np.zeros((theta.size, 3))

    # and fill in the values
    cartesian[:, 0] = np.sin(theta) * np.cos(phi)
    cartesian[:, 1] = np.sin(theta) * np.sin(phi)
    cartesian[:, 2] = np.cos(theta)

    # and we are done
    return r[:, None] * cartesian


def decay_zenith_azimuth(decay_point: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Returns the zenith angle and azimuth angle (measured from East to North) of a shower as measured at the decay point


    Parameters
    ----------
    axis: np.ndarray
        A shape=(3,) array of the geocentric x,y,z coordinates of shower axis (km)
    decay_point: np.ndarray
        A shape=(N,3) array of the geocentric x,y,z coordinates of the decay points (km).

    Returns
    -------
    theta: np.ndarray
        The zenith angle of each shower from a line normal to the Earth centered on the decay point (rad).
    phi: np.ndarray
        The azimuth angle (measured from East to North) of each shower relative to the decay point (rad).
    """

    axis = coordinates.EarthLocation(x=decay_point[:,0]+axis[0], y=decay_point[:,1]+axis[1], z=decay_point[:,2]+axis[2])
    decay_point = coordinates.EarthLocation(x=decay_point[:,0], y=decay_point[:,1], z=decay_point[:,2])
    
    enu = pm.geodetic2enu(axis.lat.value, axis.lon.value, axis.height.to(u.m).value, lat0=decay_point.lat.value, lon0=decay_point.lon.value, h0=decay_point.height.to(u.m).value)

    enu = np.array(enu).T

    rthetaphi = cartesian_to_spherical(np.array(enu))

    theta = rthetaphi[:,1]*u.rad
    phi = rthetaphi[:,2]*u.rad

    return theta, phi


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


def obs_zenith_azimuth(
    station: np.ndarray, decay_point: np.ndarray) -> np.ndarray:
    """
    Returns the zenith angle and azimuth angle (measured from East to North) at which the decay points are located relative to the station.


    Parameters
    ----------
    station: np.ndarray
        A shape=(3,) array of the geocentric x,y,z coordinates of the station (km)
    decay_point: np.ndarray
        A shape=(N,3) array of the geocentric x,y,z coordinates of the decay points (km).

    Returns
    -------
    theta: np.ndarray
        The zenith angle of each decay point from a line normal to the Earth centered on the station (rad).
    phi: np.ndarray
        The azimuth angle (measured from East to North) of each decay point relative to the station (rad).
    """

    station = coordinates.EarthLocation(x=station[0], y=station[1], z=station[2])
    decay_point = coordinates.EarthLocation(x=decay_point[:,0], y=decay_point[:,1], z=decay_point[:,2])

    enu = pm.geodetic2enu(decay_point.lat.value, decay_point.lon.value, decay_point.height.to(u.m).value, lat0=station.lat.value, lon0=station.lon.value, h0=station.height.to(u.m).value)

    enu = np.array(enu).T

    rthetaphi = cartesian_to_spherical(np.array(enu))

    theta = rthetaphi[:,1]*u.rad
    phi = rthetaphi[:,2]*u.rad

    return theta, phi 


def distance_to_horizon(height: np.ndarray, radius: float = Re) -> np.ndarray:
    """
    Calculate the distance to the horizon from a given altitude
    with a given ice thickness.

    Parameters
    ----------
    height: np.ndarray
        The payload altitude in km.
    thickness: np.ndarray
        The ice thickness in km.

    Returns
    -------
    distance: np.ndarray
        The distance to the payload in km.
    """
    return (radius + height) * np.sin(-horizon_angle(height, radius))