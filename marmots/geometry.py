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
    "points_on_earth",
    "horizon_angle",
    "rotate_around_axis",
    "find_intersection",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "project",
    "unproject",
    "geometric_area",
    "obs_zenith_azimuth",
    "decay_zenith_azimuth",
    "triangle_random_point",
    "norm",
    "normalize",
]

# create a named tuple to store our geometry information
GeometricArea = NamedTuple(
    "GeometricArea",
    [
        ("area", np.ndarray),
        ("dot", np.ndarray),
        ("emergence", np.ndarray),
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
    lat_deg: np.ndarray,
    lon_deg: np.ndarray, 
    height: np.ndarray,
    maxview: float,
    orientations: np.ndarray,
    fov: np.ndarray,
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

    # convert to radians
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    # determine the altitude of the point source given a location and time
    observing_location = EarthLocation(lat=lat_deg*u.deg, lon=lon_deg*u.deg, height=height*u.km)  
    observing_time = Time(time)  
    aa = AltAz(location=observing_location, obstime=observing_time)
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    alt = np.deg2rad(coord.transform_to(aa).alt.value)

    # determine the vector to the point source
    # the inverse of this vector is the particle axis
    source_itrs = coord.transform_to(ITRS(obstime=observing_time))
    x, y, z = source_itrs.x, source_itrs.y, source_itrs.z
    axis = -np.array([x,y,z])

    # generate N random exit points within the geometric area associated with this point source
    trials, A0, stations, orientations, fov, antennas = points_on_earth(lat, lon, height, alt, axis, maxview, orientations, fov, antennas, N, min_elev)
    
    events_generated = trials.shape[0]

    # if the Earth is not in view of any of the stations, return empty arrays and zeros.
    if (events_generated == 0):

        return GeometricArea(A0, np.array([]), np.array([]), np.array([]), np.array([]), axis, 0, np.array([]), np.array([]), np.array([]))

    else:

        # compute the dot product of each exit point with the axis vector
        dot = np.dot(normalize(trials), axis)

        # and mask those trials whose dot product < 0 - since trials are on
        # the surface of the Earth, this would require the RF to propagate
        # through the Earth which is going to render the event undetectable.
        valid = dot > 0

        # mask the dot products and view angle
        dot = dot[valid]
            
        # compute the emergence angle at each exit point
        emergence = np.pi/2.0 - np.arccos(dot) if dot.size else np.asarray([])
                   
        # and we are done
        return GeometricArea(A0, dot, emergence, stations, trials[valid], axis, events_generated, orientations, fov, antennas)


def view_angle(point: np.ndarray, obspoint: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Calculates the view angle between a point and an obersavation point, given the particle axis at the point.
    """
       
    # calculate the vector from the point to the obs. points
    view = obspoint - point

    # calculate the view angle
    return np.arccos(np.dot(normalize(view), axis))


def points_on_earth(
    lat: np.ndarray, 
    lon: np.ndarray, 
    height: np.ndarray, 
    alt: np.ndarray,  
    axis: np.ndarray, 
    maxview: float,
    orientations: np.ndarray,
    fov: np.ndarray,
    antennas: np.ndarray,
    N: int = 10_000,
    min_elev: float = np.deg2rad(-30),
) -> np.ndarray:
    """
    Sample `N` exit points, in geocentric coordinates, from the geometric area A_g associated with a point source.

    Parameters
    ----------
    lat: np.ndarray
        Latitudes of the stations, in radians.
    lon: np.ndarray
        Longitudes of the stations, in radians.
    height: np.ndarray
        Altitudes of the stations, in km.
    alt: np.ndarray
        Altitude (in radians) of the point source as measured from the position of each station.
    axis: np.ndarray
        The particle axis of a neutrino originating from the point source.
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

    Returns
    -------
    
    trials: np.ndarray
        Geocentric coordinates of the generated exit points.
    A0: float
        The geometric area A_g in which exit points are generated (km^2).
    stations: np.ndarray
        Geocentric and geodetic coordinates of the valid stations.
    orientations: np.ndarray
        Orientations of the valid stations.
    fov: np.ndarray
       The field-of-view of each valid station.
    antennas: np.ndarray
       The number of phased antennas in each valid station.
    """

    lowest_angle = alt - maxview

    horizon = horizon_angle(height, radius=Re)

    # if lowest_angle is above the horizon, this station doesn't view the Earth at all in this direction
    # also, if we're looking below our minimum elevation angle cut, cut it
    invalid = (lowest_angle > horizon) | (alt < min_elev)

    # if no stations view the Earth, exit now
    if (np.sum(invalid) == lat.size):
        trials = np.array([])
        A0 = 0 
        stations = np.array([])

    else:
        
        # save the stations that do have the Earth in view
        station_theta = np.pi/2 - lat[~invalid]
        station_phi = lon[~invalid]
        station_height = height[~invalid]

        # save the geocentric and geodetic coordinates of all valid stations
        stations = {}
        for i in range(station_theta.size):
            stations[i] = {"geocentric": spherical_to_cartesian(station_theta[i], station_phi[i], Re+station_height[i])[0], 
                            "geodetic": np.array([np.rad2deg(lat[~invalid][i]), np.rad2deg(lon[~invalid][i]), height[~invalid][i]]).T}
        orientations = orientations[~invalid]
        fov = fov[~invalid]
        antennas = antennas[~invalid]

        # Now, we must create a cone, with opening angle maxview, projected in the direction of the point source
        z = np.array([0,0,1])
        perp = np.cross(axis, z) # axis perpindicular to the shower axis
        v = rotate_around_axis(axis, perp/np.linalg.norm(perp), maxview) # the shower axis rotated by maxview
        
        theta = np.deg2rad(np.arange(0,360,15))
        vectors = np.empty((theta.size,3))
        # a cone of vectors around the shower axis, each separated by 15 degrees,
        # pointed towards the point source
        for i in range(theta.size):
               vectors[i] = -rotate_around_axis(v, axis, theta[i]) 

        # we are now going to see where this cone intersects the Earth, when projected from each station
        # as we do so, we are going to collect the sinusoidal projections of each area
        polygons = []
        for i in range(len(stations)):
            # from the vantage point of each station, see where our cone of vectors intersect the Earth
            points = cartesian_to_spherical(np.array([find_intersection(vec, stations[i]["geocentric"]) for vec in vectors]))

            # identify areas that cross the antimeridian. These points need to be moved to the same side
            s = np.sum(points[:,2] > 0)
            wrapped_over_antimeridian = (s != 0) & (s != points[:,2].size) & (np.mean(abs(points[:,2])) > np.pi/2)
            if wrapped_over_antimeridian:
                points[:,2][points[:,2] < 0] += 2*np.pi
    
            # sinusoidal projection allows us to work in 2D, while conserving the shape's area
            x,y = project(points[:,0], 90 - np.rad2deg(points[:,1]), np.rad2deg(points[:,2]))

            # this ensures that the polygon vertices are in order
            pp = list(zip(x,y))
            cent=(np.sum([p[0] for p in pp])/len(pp),np.sum([p[1] for p in pp])/len(pp))
            # sort by polar angle
            pp.sort(key=lambda p: np.arctan2(p[1]-cent[1],p[0]-cent[0]))

            polygons.append(Polygon(pp))

        
        # the union of all the areas
        multi = unary_union(polygons)

        # the union area
        # the area of A_g
        A0 = multi.area

        # triangulate the union area in order to randomly sample points uniformly
        triangles = []
        # check to see if the union area is a single polygon, or multiple polygons
        if type(multi) == shapely.geometry.multipolygon.MultiPolygon:
            # if multiple polygons:
            for i in range(len(multi.geoms)):
                vertices = np.array(multi.geoms[i].exterior.coords)[:-1]

                start = np.arange(0, vertices.shape[0])
                end = np.roll(start,-1)
                segments = np.column_stack((start,end))

                shape = dict(vertices = vertices, segments = segments)

                # the package tr performs a Delaunay Triangulation of the area
                tri = tr.triangulate(shape, 'p')

                # save all the triangle vertices
                triangles.append(tri['vertices'][tri['triangles']])
        else:
            # if a single polygon:
            vertices = np.array(multi.exterior.coords)[:-1]

            start = np.arange(0, vertices.shape[0])
            end = np.roll(start,-1)
            segments = np.column_stack((start,end))

            shape = dict(vertices = vertices, segments = segments)

            # the package tr performs a Delaunay Triangulation of the area
            tri = tr.triangulate(shape, 'p')

            # save all the triangle vertices
            triangles.append(tri['vertices'][tri['triangles']])

        # big array of triangles with shape (M,3,3), where M is the number of triangles
        triangles = np.concatenate(triangles)
        
        # find the area of each triangle
        areas = np.empty(triangles.shape[0])
        for i in range(triangles.shape[0]):
            areas[i] = Polygon(triangles[i]).area

        # randomly pick triangles, weighed by their area 
        idx = np.random.choice(np.arange(triangles.shape[0]), p=areas/np.sum(areas), size=N) 
        # count how many times each triangle was picked
        idx, counts = np.unique(idx, return_counts = True)
        # and sample that many points from each triangle 
        proj_trials = np.concatenate(list(map(triangle_random_point, triangles[idx], counts))) 
        
        # reverse sinusoidal projection
        trials_spherical = unproject(proj_trials, Re)

        # calculate the geocentric coordinates of the exit points
        trials = spherical_to_cartesian(trials_spherical[:,1], trials_spherical[:,2], trials_spherical[:,0])

    # and return the trials, area, and valid stations
    return trials, A0, stations, orientations, fov, antennas


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

    return zenith, azimuth, decay_point_spherical
    


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
    
    y = np.sin(lon - decay_point_spherical[:,2]) * np.cos(np.pi/2 - decay_point_spherical[:,1])
    x = np.cos(lat) * np.sin(np.pi/2 - decay_point_spherical[:,1]) - np.sin(lat) * np.cos(np.pi/2 - decay_point_spherical[:,1]) * np.cos(lon - decay_point_spherical[:,2])
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
