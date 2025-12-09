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

from astropy.coordinates import EarthLocation, SkyCoord, ITRS
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import AltAz

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
    time: str = '2025-03-20 12:00:00'
):
    """
    Compute the geometric area on the surface of the Earth "illuminated"
    within a maximum `view` angle in the direction of a point source at ra_deg and dec_deg,
    from each station.

    Parameters
    ----------
    ra_deg: float
        Right ascension of the point source, in degrees.
    dec_deg: float
        Declination of the point source, in degrees.
    lat_deg: np.ndarray
        Latitudes of the stations, in degrees.
    lon_deg: np.ndarray
        Longitudes of the stations, in degrees.
    height: np.ndarray
        Altitudes of the stations, in km.
    maxview: float
        Opening angle of the cone projected in the direction of the point source, in radians.
    orientations: np.ndarray
       The orientation of each station (in degrees, relative to east).
    fov: np.ndarray
       The field-of-view of each station (in degrees).
    antennas: np.ndarray
       The number of phased antennas in each station.
    N: int
        The number of trials (exit points) to generate.
    min_elev: float
        Elevation angles below this threshold will not be simulated. Effective area will be assumed to be zero.
    time: str
        The time in which to calculate the instantaneous effective area.

    Returns
    -------
    area: float
        The geometric area A_g in which exit points are generated (km^2).
    dot: np.ndarray
        The dot product between exit point location and particle axis.
    emergence: np.ndarray
        The emergence angles for each trial (radians).
    stations: np.ndarray
        Geocentric and geodetic coordinates of the valid stations.
    trials: np.ndarray
        Geocentric coordinates of the generated exit points.
    axis: np.ndarray
        Particle axis unit vector.
    N: float
        The number of trials generated.
    orientations: np.ndarray
        Orientations of the valid stations.
    fov: np.ndarray
       The field-of-view of each valid station.
    antennas: np.ndarray
       The number of phased antennas in each valid station.
    
    """

    lat = TotalArea.stations_geodetic[:,0]
    lon = TotalArea.stations_geodetic[:,1]
    height = TotalArea.stations_geodetic[:,2]

    # determine the altitude of the point source given a location and time
    observing_location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=height*u.m)  
    observing_time = Time(time)  
    aa = AltAz(location=observing_location, obstime=observing_time)
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    alt = np.deg2rad(coord.transform_to(aa).alt.value)
    
    # check to see that we're looking above the minimum elevation angle
    above_cut = alt >= min_elev
    
    stations_geocentric = TotalArea.stations_geocentric[above_cut]
    stations_geodetic = TotalArea.stations_geodetic[above_cut]
    orientations = TotalArea.orientations[above_cut]
    fov = TotalArea.fov[above_cut]
    antennas = antennas[above_cut]

    # determine the vector to the point source
    # the inverse of this vector is the particle axis
    source_itrs = coord.transform_to(ITRS(obstime=observing_time))
    x, y, z = source_itrs.x, source_itrs.y, source_itrs.z
    axis = -np.array([x,y,z])
    
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

        return GeometricArea(0, np.array([]), np.array([]), np.array([]), np.array([]), axis, 0, np.array([]), np.array([]), np.array([]))

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


def view_angle(point: np.ndarray, obspoint: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Calculates the view angle between a point and an obersavation point, given the particle axis at the point.
    """
       
    # calculate the vector from the point to the obs. points
    view = obspoint - point

    # calculate the view angle
    return np.arccos(np.dot(normalize(view), axis))


def triangle_random_point(triangle, size):
    """
    Randomly samples 'size' number of points from a 3D triangle defined by its three vertices.
    """
    r1 = np.random.random(size)
    r2 = np.random.random(size)

    P = (1 - np.sqrt(r1)).reshape(-1,1) * triangle[0] + (np.sqrt(r1) * (1 - r2)).reshape(-1,1) * triangle[1] + (np.sqrt(r1) * r2).reshape(-1,1) * triangle[2]
    
    return P


def rotate_around_axis(vector, axis, theta):
    """
    Rotates a vector around 'axis' by 'theta' (in radians).
    """
    R = np.empty((3,3))
    R[0] = np.array([np.cos(theta)+axis[0]**2 * (1-np.cos(theta)), axis[0]*axis[1]*(1-np.cos(theta)) - axis[2]*np.sin(theta),axis[0]*axis[2]*(1-np.cos(theta)) + axis[1]*np.sin(theta)])
    R[1] = np.array([axis[1]*axis[0]*(1-np.cos(theta)) + axis[2]*np.sin(theta),np.cos(theta)+axis[1]**2 * (1-np.cos(theta)),axis[1]*axis[2]*(1-np.cos(theta)) - axis[0]*np.sin(theta)])
    R[2] = np.array([axis[2]*axis[0]*(1-np.cos(theta)) - axis[1]*np.sin(theta),axis[2]*axis[1]*(1-np.cos(theta)) + axis[0]*np.sin(theta),np.cos(theta)+axis[2]**2 * (1-np.cos(theta))])
    
    return R @ vector


def find_intersection(vector, station):
    """
    See: https://diegoinacio.github.io/computer-vision-notebooks-page/pages/ray-intersection_sphere.html

    Calculates where a vector from a station intersects a sphere with Earth's radius.
    
    note: this assumes that the vector is pointed toward the source
    """
    t = np.dot(station, vector)
    p = station - vector*t
    d = np.linalg.norm(p)

    height = np.linalg.norm(station)
    horizon_elev = -np.arccos(Re / height) # the angle from horizontal to the horizon from the station
    vec_elev = -np.arccos(np.dot(station/height,vector)) + np.pi/2 # the angle from horizontal for the observation vector

    if(vec_elev > horizon_elev):
        # if the vector points above the horizon, rotate it towards the Earth such that it hits the horizon
        # this way we do not sample points beyond the visible horizon
        axis = np.cross(station/height, vector) # axis perpendicular to the station vector and observation vector
        horizon_vec = rotate_around_axis(vector, axis, -(horizon_elev-vec_elev)) # rotate the observation vector such that it is pointed at the horizon
        new_t = np.dot(station, horizon_vec)
        new_p = station - horizon_vec*new_t
        Ps = new_p

    elif(vec_elev == horizon_elev):
        # this happens when the vector is tangent to the Earth
        Ps = p

    else:
        # find the first point of intersection when the vector passes through the Earth
        i = np.sqrt(Re**2 - d**2)
        Ps = station - vector*(t + i)
        
    return Ps


def cartesian_to_spherical(point):
    """
    Converts 3D cartesian coordinates to spherical coordinates.
    """
    spherical = np.empty((point.shape[0], 3))

    spherical[:,0] = norm(point)
    spherical[:,1] = np.arccos(point[:,2]/norm(point))
    spherical[:,2] = np.arctan2(point[:,1], point[:,0])

    return spherical


def project(radius, latitude, longitude):
    """
    Performs sinusoidal projection given a sphere's radius, and the latitude and longitude of the point.
    """
    lat_dist = np.pi * radius / 180
    y = latitude * lat_dist 
    x = longitude * lat_dist * np.cos(np.deg2rad(latitude))
    return x, y


def unproject(point, radius):
    """
    Reverse sinusoidal projection.
    """
    spherical = np.empty((point.shape[0], 3))
    
    lat_dist = np.pi * radius/180
    latitude = point[:,1]/lat_dist
    longitude = point[:,0]/lat_dist/np.cos(np.deg2rad(latitude))
    
    spherical[:,0] = radius
    spherical[:,1] = np.deg2rad(90 - latitude)
    spherical[:,2] = np.deg2rad(longitude)
    return spherical


def horizon_angle(height: np.ndarray, radius: float = Re) -> np.ndarray:
    """
    Calculate the horizon angle (in radians)
    for a given set of heights (in km).

    Parameters
    ----------
    height: np.ndarray
       The height of the viewing point (km).
    radius: float
       The Earth radius to use (in radians).

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
    zenith: np.ndarray
        The zenith angle of each shower from a line normal to the Earth centered on the decay point (rad).
    azimuth: np.ndarray
        The azimuth angle (measured from East to North) of each shower relative to the decay point (rad).
    decay_point_spherical: np.ndarray
        Spherical coordinates of the decay point.
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
    zenith: np.ndarray
        The zenith angle of each decay point from a line normal to the Earth centered on the station (rad).
    azimuth: np.ndarray
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


@njit
def norm(vec: np.ndarray):
    """
    Calculates the magnitude of an array of vectors along axis=1. Faster than using np.linalg.norm(axis=1) for very large arrays.
    """
    # norm along axis=1
    return np.sqrt(vec[:,0]**2 +vec[:,1]**2 + vec[:,2]**2)


@njit
def normalize(vec: np.ndarray):
    """
    Normalizes an array of vectors along axis=1. Faster than using np.linalg.norm(axis=1) for very large arrays.
    """
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
