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
    "emergence_angle",
    "decay_altitude",
    "triangle_random_point",
    "norm",
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
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    alt = altitude(ra, dec, lat, lon)

    theta = np.pi/2 - dec
    phi = ra  

    # the particle axis
    axis = -spherical_to_cartesian(theta, phi, r=1.0)[0]

    # generate ~N random points in the corresponding segment
    trials, A0, stations, orientations, fov = points_on_earth(lat, lon, height, alt, axis, maxview, orientations, fov, N, min_elev)
    
    events_generated = trials.shape[0]
    
    if (events_generated == 0):

        return GeometricArea(A0, np.array([]), np.array([]), np.array([]), np.array([]), axis, 0, np.array([]), np.array([]))

    else:

        '''
        in_view = np.zeros(events_generated)
        for i in range(stations["geocentric"].shape[0]):
            in_view += view_angle(trials, stations["geocentric"][i], axis) <= maxview 
            
        trials = trials[in_view > 0]
        '''

        # compute the dot product of each trial point with the axis vector
        dot = np.dot(normalize(trials), axis)

        # and mask those trials whose dot product < 0 - since trials are on
        # the surface of the Earth, this would require the RF to propagate
        # through the Earth which is going to render the event undetectable.
        valid = dot > 0

        # mask the dot products and view angle
        dot = dot[valid]
            
        # compute the average emergence angle for this elevation
        emergence = np.pi/2.0 - np.arccos(dot) if dot.size else np.asarray([])
                   
        # and we are done
        return GeometricArea(A0, dot, emergence, stations, trials[valid], axis, events_generated, orientations, fov)


def altitude(ra, dec, lat, lon):
    alt = np.arcsin( np.sin(dec)*np.sin(lat) + np.cos(dec)*np.cos(lat)*np.cos(lon-ra) )
    return alt


def decay_view(
    decay_point: np.ndarray, axis: np.ndarray, station: np.ndarray,
) -> np.ndarray:

    d = station - decay_point
    
    return np.arccos(np.dot(normalize(d), axis)) 


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


def points_on_earth(
    lat: np.ndarray, 
    lon: np.ndarray, 
    height: np.ndarray, 
    alt: np.ndarray,  
    axis: np.ndarray, 
    maxview: float,
    orientations: np.ndarray,
    fov: np.ndarray,
    N: int = 10_000,
    min_elev: float = np.deg2rad(-30),
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
        
        stations = {}
        for i in range(station_theta.size):
            stations[i] = {"geocentric": spherical_to_cartesian(station_theta[i], station_phi[i], Re+station_height[i])[0], 
                            "geodetic": np.array([np.rad2deg(lat[~invalid][i]), np.rad2deg(lon[~invalid][i]), height[~invalid][i]]).T}
        orientations = orientations[~invalid]
        fov = fov[~invalid]
        
        
        z = np.array([0,0,1])
        perp = np.cross(axis, z) # axis perpindicular to the shower axis
        v = rotate_around_axis(axis, perp/np.linalg.norm(perp), maxview) # the shower axis rotated by maxview
        
        theta = np.deg2rad(np.arange(0,360,15))
        vectors = np.empty((theta.size,3))
        for i in range(theta.size):
               vectors[i] = -rotate_around_axis(v, axis, theta[i]) # a cone of vectors around the shower axis
        
        polygons = []
        for i in range(len(stations)):
            # from the vantage point of each station, see where our cone of vectors intersect the Earth
            points = cartesian_to_spherical(np.array([find_intersection(vec, stations[i]["geocentric"]) for vec in vectors]))

            # identify areas that cross on the antimeridian. These points need to be moved all to the same side
            s = np.sum(points[:,2] > 0)
            wrapped_over_antimeridian = (s != 0) & (s != points[:,2].size) & (np.mean(abs(points[:,2])) > np.pi/2)
            if wrapped_over_antimeridian:
                points[:,2][points[:,2] < 0] += 2*np.pi
    
            # sinusoidal projection allows us to work in 2D, while conserving the shape's area
            x,y = project(points[:,0], 90 - np.rad2deg(points[:,1]), np.rad2deg(points[:,2]))

            # this ensures that the vertices are in order
            pp = list(zip(x,y))
            cent=(np.sum([p[0] for p in pp])/len(pp),np.sum([p[1] for p in pp])/len(pp))
            # sort by polar angle
            pp.sort(key=lambda p: np.arctan2(p[1]-cent[1],p[0]-cent[0]))

            polygons.append(Polygon(pp))

        
        # the union of all the areas
        multi = unary_union(polygons)

        # the union area
        A0 = multi.area

        # triangulate the union area in order to randomly sample points uniformly
        triangles = []
        if type(multi) == shapely.geometry.multipolygon.MultiPolygon:
            for i in range(len(multi.geoms)):
                vertices = np.array(multi.geoms[i].exterior.coords)[:-1]

                start = np.arange(0, vertices.shape[0])
                end = np.roll(start,-1)
                segments = np.column_stack((start,end))

                shape = dict(vertices = vertices, segments = segments)

                tri = tr.triangulate(shape, 'p')

                triangles.append(tri['vertices'][tri['triangles']])
        else:
            vertices = np.array(multi.exterior.coords)[:-1]

            start = np.arange(0, vertices.shape[0])
            end = np.roll(start,-1)
            segments = np.column_stack((start,end))

            shape = dict(vertices = vertices, segments = segments)

            tri = tr.triangulate(shape, 'p')

            triangles.append(tri['vertices'][tri['triangles']])

        triangles = np.concatenate(triangles)
        
        # find the area of each triangle
        areas = np.empty(triangles.shape[0])
        for i in range(triangles.shape[0]):
            areas[i] = Polygon(triangles[i]).area
            
        idx = np.random.choice(np.arange(triangles.shape[0]), p=areas/np.sum(areas), size=N) # randomly pick triangles weighed by their area 
        idx, counts = np.unique(idx, return_counts = True) # count how many times each triangle was picked
        proj_trials = np.concatenate(list(map(triangle_random_point, triangles[idx], counts))) # and sample that many points from each triangle 
        
        # reverse sinusoidal projection
        trials_spherical = unproject(proj_trials)

        trials = spherical_to_cartesian(trials_spherical[:,1], trials_spherical[:,2], trials_spherical[:,0])

    # and return the trials, area, and valid stations
    return trials, A0, stations, orientations, fov


def triangle_random_point(triangle, size):
    r1 = np.random.random(size)
    r2 = np.random.random(size)

    P = (1 - np.sqrt(r1)).reshape(-1,1) * triangle[0] + (np.sqrt(r1) * (1 - r2)).reshape(-1,1) * triangle[1] + (np.sqrt(r1) * r2).reshape(-1,1) * triangle[2]
    
    return P


def rotate_around_axis(vector, axis, theta):
    R = np.empty((3,3))
    R[0] = np.array([np.cos(theta)+axis[0]**2 * (1-np.cos(theta)), axis[0]*axis[1]*(1-np.cos(theta)) - axis[2]*np.sin(theta),axis[0]*axis[2]*(1-np.cos(theta)) + axis[1]*np.sin(theta)])
    R[1] = np.array([axis[1]*axis[0]*(1-np.cos(theta)) + axis[2]*np.sin(theta),np.cos(theta)+axis[1]**2 * (1-np.cos(theta)),axis[1]*axis[2]*(1-np.cos(theta)) - axis[0]*np.sin(theta)])
    R[2] = np.array([axis[2]*axis[0]*(1-np.cos(theta)) - axis[1]*np.sin(theta),axis[2]*axis[1]*(1-np.cos(theta)) + axis[0]*np.sin(theta),np.cos(theta)+axis[2]**2 * (1-np.cos(theta))])
    
    return R @ vector


def find_intersection(vector, station):
    
    #https://diegoinacio.github.io/computer-vision-notebooks-page/pages/ray-intersection_sphere.html
    
    #note: this assumes that the vector is pointed toward the source
    
    t = np.dot(station, vector)
    p = station - vector*t
    d = np.linalg.norm(p)

    height = np.linalg.norm(station)
    horizon_elev = -np.arccos(Re / height) # the angle from horizontal to the horizon from the station
    vec_elev = -np.arccos(np.dot(station/height,vector)) + np.pi/2 # the angle from horizontal for the observation vector

    if(vec_elev > horizon_elev):
        # if the vector points above the horizon, rotate it towards the Earth such that it hits the horizon
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

    spherical = np.empty((point.shape[0], 3))

    spherical[:,0] = norm(point)
    spherical[:,1] = np.arccos(point[:,2]/norm(point))
    spherical[:,2] = np.arctan2(point[:,1], point[:,0])

    return spherical


def project(radius, latitude, longitude):
    lat_dist = np.pi * radius / 180
    y = latitude * lat_dist 
    x = longitude * lat_dist * np.cos(np.deg2rad(latitude))
    return x, y


def unproject(point):
    spherical = np.empty((point.shape[0], 3))
    
    lat_dist = np.pi * Re/180
    latitude = point[:,1]/lat_dist
    longitude = point[:,0]/lat_dist/np.cos(np.deg2rad(latitude))
    
    spherical[:,0] = Re
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
    ice: float
       The constant ice thickness (km)

    Returns
    -------
    horizon: np.ndarray
       The horizon angle (in radians).
    """
    return -np.arccos((radius) / (radius + height))


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

    return zenith, azimuth, decay_point_spherical


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
    
    y = np.sin(lon - decay_point_spherical[:,2]) * np.cos(np.pi/2 - decay_point_spherical[:,1])
    x = np.cos(lat) * np.sin(np.pi/2 - decay_point_spherical[:,1]) - np.sin(lat) * np.cos(np.pi/2 - decay_point_spherical[:,1]) * np.cos(lon - decay_point_spherical[:,2])
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
