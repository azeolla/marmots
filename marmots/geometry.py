"""
This module provides free-functions for calculating various
needed geometric quantities and formulas.
"""
from typing import Any, NamedTuple, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
import triangle as tr

from marmots.constants import Re

from numba import jit, njit

__all__ = [
    "view_angle",
    "altitude",
    "horizon_angle",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "geometric_area",
    "obs_zenith_azimuth",
    "decay_zenith_azimuth",
    "decay_altitude",
    "triangle_random_point",
    "norm",
    "exit_zenith"
]

GeometricArea = NamedTuple(
    "GeometricArea",
    [
        ("area", np.ndarray),
        ("mesh", np.ndarray),
        ("dot", np.ndarray),
        ("stations", np.ndarray),
        ("trials", np.ndarray),
        ("axis", np.ndarray),
        ("N", float),
        ("orientations", np.ndarray),
        ("fov", np.ndarray),
        ("antennas", np.ndarray),
    ],
)


def geometric_area(
    ra_deg: float,
    dec_deg: float,
    TotalArea: NamedTuple,
    maxview: float,
    antennas: np.ndarray,
    N: int = 10_000,
    min_elev: float = np.deg2rad(-30),
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

    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    alt = altitude(ra, dec, np.deg2rad(TotalArea.stations_geodetic[:,0]), np.deg2rad(TotalArea.stations_geodetic[:,1]))
    
    # check to see that we're looking above the minimum elevation angle
    above_cut = alt >= min_elev
    
    stations_geocentric = TotalArea.stations_geocentric[above_cut]
    stations_geodetic = TotalArea.stations_geodetic[above_cut]
    orientations = TotalArea.orientations[above_cut]
    fov = TotalArea.fov[above_cut]
    antennas = antennas[above_cut]

    theta = np.pi/2 - dec
    phi = ra  

    # the particle axis
    axis = -spherical_to_cartesian(theta, phi, r=1.0)[0]
    
    in_view = np.zeros(TotalArea.centroids.shape[0], dtype=bool)
    valid = np.zeros(stations_geocentric.shape[0], dtype=bool)
    for i in range(stations_geocentric.shape[0]):
        view = view_angle(TotalArea.centroids, stations_geocentric[i], axis) <= maxview 
        in_view[view] = 1
        valid[i] = np.sum(view) > 0
    
    surface = TotalArea.triangles[in_view]
    
    stations_geocentric = stations_geocentric[valid]
    stations_geodetic = stations_geodetic[valid]
    orientations = orientations[valid]
    fov = fov[valid]
    antennas = antennas[valid]
    
    if (surface.shape[0] == 0):

        return GeometricArea(0, np.array([]), np.array([]), np.array([]), np.array([]), axis, 0, np.array([]), np.array([]))

    else:
        
        A, normals = area(surface)
        A0 = np.sum(A)

        # compute the dot product of each trial point with the axis vector
        tdot = np.dot(normals, axis)
        
        idx = np.random.choice(np.arange(surface.shape[0]), p=A/np.sum(A), size=N) # randomly pick triangles weighed by their area 
        idx, counts = np.unique(idx, return_counts = True) # count how many times each triangle was picked
        trials = np.concatenate(list(map(triangle_random_point, surface[idx], counts))) # and sample that many points from each triangle
        dot = np.repeat(tdot[idx], counts)

        # and mask those trials whose dot product < 0 - since trials are on
        # the surface of the Earth, this would require the RF to propagate
        # through the Earth which is going to render the event undetectable.
        out_earth = dot > 0

        # mask the dot products and view angle
        dot = dot[out_earth]
        
        stations = {}
        for j in range(stations_geocentric.shape[0]):
            stations[j] = {"geocentric": stations_geocentric[j], 
                            "geodetic": stations_geodetic[j]}
                   
        # and we are done
        return GeometricArea(A0, surface, dot, stations, trials[out_earth], axis, trials.shape[0], orientations, fov, antennas)


def decay_view(
    decay_point: np.ndarray, axis: np.ndarray, station: np.ndarray,
) -> np.ndarray:

    d = station - decay_point
    
    return np.arccos(np.dot(normalize(d), axis)) 


def altitude(ra, dec, lat, lon):
    alt = np.arcsin( np.sin(dec)*np.sin(lat) + np.cos(dec)*np.cos(lat)*np.cos(lon-ra) )
    return alt


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
    view = obspoint - point

    # calculate the view angle
    return np.arccos(np.dot(normalize(view), axis))


def triangle_random_point(triangle, size):
    r1 = np.random.random(size)
    r2 = np.random.random(size)

    P = (1 - np.sqrt(r1)).reshape(-1,1) * triangle[0] + (np.sqrt(r1) * (1 - r2)).reshape(-1,1) * triangle[1] + (np.sqrt(r1) * r2).reshape(-1,1) * triangle[2]
    
    return P


def cartesian_to_spherical(point):

    spherical = np.empty((point.shape[0], 3))

    spherical[:,0] = norm(point)
    spherical[:,1] = np.arccos(point[:,2]/norm(point))
    spherical[:,2] = np.arctan2(point[:,1], point[:,0])

    return spherical


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
    cartesian = np.empty((theta.size, 3))

    # and fill in the values
    cartesian[:, 0] = r * np.sin(theta) * np.cos(phi)
    cartesian[:, 1] = r * np.sin(theta) * np.sin(phi)
    cartesian[:, 2] = r * np.cos(theta)

    # and we are done
    return cartesian


def exit_zenith(exit_point: np.ndarray, axis: np.ndarray):
    
    dot = np.dot(normalize(exit_point), axis)
    zenith = np.arccos(dot)
    
    return zenith


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

    dot1 = np.dot(normalize(decay_point), axis)
    zenith = np.arccos(dot1)
    
    decay_point_spherical = cartesian_to_spherical(decay_point)
    axis_spherical = cartesian_to_spherical(axis[None,:])
    
    y = np.sin(decay_point_spherical[:,2] - axis_spherical[:,2]) * np.cos(np.pi/2 - axis_spherical[:,1])
    x = np.cos(np.pi/2 - decay_point_spherical[:,1]) * np.sin(np.pi/2 - axis_spherical[:,1]) - np.sin(np.pi/2 - decay_point_spherical[:,1]) * np.cos(np.pi/2 - axis_spherical[:,1]) * np.cos(decay_point_spherical[:,2] - axis_spherical[:,2])
    a = np.arctan2(y,x)
    
    azimuth = a + np.pi/2

    return zenith, azimuth


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
    station: np.ndarray, decay_point: np.ndarray, decay_point_spherical: np.ndarray) -> np.ndarray:
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

    decay_vector = decay_point - station['geocentric']
    decay_vector = normalize(decay_vector)

    dot1 = np.dot(decay_vector, station['geocentric']/np.linalg.norm(station['geocentric']))
    zenith = np.arccos(dot1)

    lat = np.deg2rad(station['geodetic'][0])
    lon = np.deg2rad(station['geodetic'][1])
    
    dlat = np.deg2rad(decay_point_spherical[:,0])
    dlon = np.deg2rad(decay_point_spherical[:,1])
    
    y = np.sin(lon - dlon) * np.cos(dlat)
    x = np.cos(lat) * np.sin(dlat) - np.sin(lat) * np.cos(dlat) * np.cos(lon - dlon)
    a = np.arctan2(y,x)
    
    # first add pi/2 so that azimuth is measured from East instead of North. Then wrap azimuth between [-pi,pi)
    azimuth = ((a + np.pi/2) + np.pi) % (2*np.pi) - np.pi

    return zenith, azimuth


def distance_to_horizon(height: float, radius: float = Re) -> float:
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
    return (radius + height) * np.sin(-horizon_angle(height))


@njit
def norm(vec: np.ndarray):
    # norm along axis=1
    return np.sqrt(vec[:,0]**2 +vec[:,1]**2 + vec[:,2]**2)


@njit
def normalize(vec: np.ndarray):
    # normalize along axis=1
    norm = np.sqrt(vec[:,0]**2 +vec[:,1]**2 + vec[:,2]**2)
    return vec/np.expand_dims(norm, 1)


def normal(triangles):
    # The cross product of two sides is a normal vector
    return np.cross(triangles[:,1] - triangles[:,0], 
                    triangles[:,2] - triangles[:,0], axis=1)

def area(triangles):
    # The norm of the cross product of two sides is twice the area
    n = normal(triangles)
    mag = norm(n)
    area = mag/2
    
    return area, n/mag[:, None]
