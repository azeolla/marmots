"""
This module provides free-functions for calculating various
needed geometric quantities and formulas.
"""
from shutil import ReadError
from typing import Any, NamedTuple, Tuple

import numpy as np
import astropy.units as u
import astropy.coordinates as coordinates
import shapely
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, split, transform
from math import remainder

from marmots.constants import Re
import pymap3d as pm

from numba import jit, njit
import numba

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
    az = azimuth(ra, dec, lat, lon)

    theta = np.pi/2 - dec
    phi = ra  

    # the particle axis
    axis = -spherical_to_cartesian(theta, phi, r=1.0)[0]

    # generate ~N random points in the corresponding segment
    trials, A0, stations, orientations, fov = points_on_earth(lat, lon, height, alt, az, maxview, orientations, fov, N)
    
    events_generated = trials.shape[0]
    
    if (events_generated == 0):

        return GeometricArea(A0, np.array([]), np.array([]), np.array([]), np.array([]), axis, 0, np.array([]), np.array([]))

    else:

        in_view = np.zeros(events_generated)
        for i in range(stations["geocentric"].shape[0]):
            in_view += view_angle(trials, stations["geocentric"][i], axis) <= maxview 
            
        trials = trials[in_view > 0]

        # compute the dot product of each trial point with the axis vector
        dot = np.dot(trials/np.linalg.norm(trials, axis=1)[:,None], axis)

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


def azimuth(ra, dec, lat, lon):
    az = np.arctan2(np.sin(lon-ra)  , (np.cos(lon-ra)*np.sin(lat) - np.tan(dec)*np.cos(lat)))
    return az + np.pi


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
    view = obspoint- point

    # and make sure that view is normalized
    view = view/np.linalg.norm(view, axis=1)[:,None]

    # make sure that axis is normalized (just in case)
    # NB: don't use /= as it will overwrite the argument
    axis = axis / np.linalg.norm(axis)

    # calculate the view angle
    return np.arccos(np.dot(view, axis))


@jit(nopython=True)
def inside_poly(polygon, point):
  length = len(polygon)-1
  dy2 = point[1] - polygon[0][1]
  intersections = 0
  ii = 0
  jj = 1

  while ii<length:
    dy  = dy2
    dy2 = point[1] - polygon[jj][1]

    # consider only lines which are not completely above/bellow/right from the point
    if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):
        
      # non-horizontal line
      if dy<0 or dy2<0:
        F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

        if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
          intersections += 1
        elif point[0] == F: # point on line
          return 2

      # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
      elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
        return 2

      # there is another posibility: (dy=0 and dy2>0) or (dy>0 and dy2=0). It is skipped 
      # deliberately to prevent break-points intersections to be counted twice.
    
    ii = jj
    jj += 1
            
  #print 'intersections =', intersections
  return intersections & 1  


@njit(parallel=True)
def inside_poly_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean) 
    for i in numba.prange(ln):
        D[i] = inside_poly(polygon,points[i])
    return D  


def points_on_earth(
    lat: np.ndarray, 
    lon: np.ndarray, 
    height: np.ndarray, 
    alt: np.ndarray,  
    az: np.ndarray, 
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

    minview = alt + view
    maxview = alt - view

    horizon = horizon_angle(height, radius=Re)

    # if minview is above the horizon, it should be set to the horizon
    minview[minview > horizon] = horizon[minview > horizon]

    # if maxview is above the horizon, we don't view the Earth at all
    minview[maxview > horizon] = np.nan
    maxview[maxview > horizon] = np.nan

    minaz = az - view
    maxaz = az + view

    # if we're looking closing to straight down then maxview will be greater than 90 deg.
    
    maxview[maxview < -np.pi/2] = -np.pi/2

    # find the indices in which the Earth is not in sight
    invalid = (np.isnan(minview) | np.isnan(maxview))

    if (np.sum(invalid) == lat.size):
        trials = np.array([])
        A0 = 0 
        stations = np.array([])

    else:
        # get the geocentric angle formed by these view angles
        theta_min = geocentric_angle(height[~invalid], maxview[~invalid], radius=Re)
        theta_max = geocentric_angle(height[~invalid], minview[~invalid], radius=Re)
        phi_min = -minaz[~invalid] # negative sign to account for azimuth being positive towards the East
        phi_max = -maxaz[~invalid]

        # draw out the polygons formed by these angles
        diff = theta_max - theta_min
        point1 = spherical_to_cartesian(theta_min, phi_min, r=Re)
        point2 = spherical_to_cartesian(theta_min + diff/4, phi_min, r=Re)
        point3 = spherical_to_cartesian(theta_min + diff/2, phi_min, r=Re)
        point4 = spherical_to_cartesian(theta_min + 3*diff/4, phi_min, r=Re)
        point5 = spherical_to_cartesian(theta_max, phi_min, r=Re)
        point6 = spherical_to_cartesian(theta_max, phi_max, r=Re)
        point7 = spherical_to_cartesian(theta_max - diff/4, phi_max, r=Re)
        point8 = spherical_to_cartesian(theta_max - diff/2, phi_max, r=Re)
        point9 = spherical_to_cartesian(theta_max - 3*diff/4, phi_max, r=Re)
        point10 = spherical_to_cartesian(theta_min, phi_max, r=Re)

        points = np.array(list(zip(point1 , point2, point3, point4, point5, point6, point7, point8, point9, point10)))

        station_theta = np.pi/2 - lat[~invalid]
        station_phi = lon[~invalid]
        station_height = height[~invalid]

        # rotate the area vertices to their appropriate place on Earth
        rotated = np.array(list(map(rotate_earth, points, station_theta, station_phi)))

        # convert the vertices back to spherical
        rthetaphi = np.array(list(map(cartesian_to_spherical, rotated)))

        # find the vertices in which some have negative phi and some have postive phi (the shape passes over the prime or anti meridian)
        sum = (np.sum(rthetaphi[:,:,2] > 0, axis=1))
        # the shapes wrap correctly over the prime meridian (phi = 0), but mess up at the antimeridian (phi = pi)
        wrapped_over_antimeridian = rthetaphi[(sum != 0) & (sum != 10) & (np.mean(abs(rthetaphi[:,:,2]), axis=1) > np.pi/2)]
        negative_phi = wrapped_over_antimeridian[:,:,2] < 0
        wrapped_over_antimeridian[:,:,2][negative_phi] += 2*np.pi # add 2pi to the negative phi values
        rthetaphi[(sum != 0) & (sum != 10) & (np.mean(abs(rthetaphi[:,:,2]), axis=1) > np.pi/2)] = wrapped_over_antimeridian

        # perform a sinuisodal projection on the vertices 
        projection = np.swapaxes(np.array(list(map(project, rthetaphi[:,:,0], 90 - np.rad2deg(rthetaphi[:,:,1]), np.rad2deg(rthetaphi[:,:,2])))), 1, 2)
        
        # sometimes the vertices aren't listed in sequential order. This will sort them
        ordered_points = []
        for i in range(projection.shape[0]):
            pp = projection[i].tolist()
            cent=(np.sum([p[0] for p in pp])/len(pp),np.sum([p[1] for p in pp])/len(pp))
            # sort by polar angle
            pp.sort(key=lambda p: np.arctan2(p[1]-cent[1],p[0]-cent[0]))
            ordered_points.append(pp)

        # and get the total geometric area
        A0, polygon = union_area(ordered_points)
        
        # generate the exit points (trials)
        if type(polygon) == shapely.geometry.multipolygon.MultiPolygon:
            proj_trials = []
            for i in range(len(polygon.geoms)):
                num = np.rint(N * polygon.geoms[i].area/A0).astype(int) 
                points = []
                minx, miny, maxx, maxy = polygon.geoms[i].bounds
                poly = np.array(list(polygon.geoms[i].exterior.coords))
                while len(points) < num:
                    x, y = np.random.uniform(minx, maxx, num), np.random.uniform(miny, maxy, num)
                    point = np.array(list(zip(x,y)))
                    contains = inside_poly_parallel(point, poly)
                    points += np.array(point)[contains].tolist()

                proj_trials += np.array(points)[:num].tolist()
            proj_trials = np.array(proj_trials)
            
        else:
            points = []
            minx, miny, maxx, maxy = polygon.bounds
            poly = np.array(list(polygon.exterior.coords))
            while len(points) < N:
                x, y = np.random.uniform(minx, maxx, N), np.random.uniform(miny, maxy, N)
                point = np.array(list(zip(x,y)))
                contains = inside_poly_parallel(point, poly)
                points += np.array(point)[contains].tolist()

            proj_trials = np.array(points)[:N]

        trials_spherical = unproject(proj_trials)

        trials = spherical_to_cartesian(trials_spherical[:,1], trials_spherical[:,2], trials_spherical[:,0])

        # the cartesian coordinates of the stations with land in view
        stations = {"geocentric": spherical_to_cartesian(station_theta, station_phi, Re+station_height), 
                    "geodetic": np.array([np.rad2deg(lat[~invalid]), np.rad2deg(lon[~invalid]), height[~invalid]]).T}
        orientations = orientations[~invalid]
        fov = fov[~invalid]

    # and return the trials, area, and valid stations
    return trials, A0, stations, orientations, fov


def rotate_earth(point, theta, phi):

    theta = -theta
    phi += np.pi

    Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

    Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

    return np.einsum('ij, ...j->...i', Rz, np.einsum('ij, ...j->...i', Ry, point))


def cartesian_to_spherical(point):

    spherical = np.zeros((point.shape[0], 3))

    spherical[:,0] = np.linalg.norm(point, axis=1)
    spherical[:,1] = np.arccos(point[:,2]/np.linalg.norm(point, axis=1))
    spherical[:,2] = np.arctan2(point[:,1], point[:,0])

    return spherical


def project(radius, latitude, longitude):
    lat_dist = np.pi * radius / 180
    y = latitude * lat_dist 
    x = longitude * lat_dist * np.cos(np.deg2rad(latitude))
    return x, y


def unproject(point):
    spherical = np.zeros((point.shape[0], 3))
    
    lat_dist = np.pi * Re/180
    latitude = point[:,1]/lat_dist
    longitude = point[:,0]/lat_dist/np.cos(np.deg2rad(latitude))
    
    spherical[:,0] = Re
    spherical[:,1] = np.deg2rad(90 - latitude)
    spherical[:,2] = np.deg2rad(longitude)
    return spherical


def union_area(
    points: list,
):
    # find the union of all the polygons
    polygon = unary_union([Polygon(x) for x in points])

    # define the the boundary of the sinusoidal projection (the antimeridian)
    left_bound = project(Re, np.arange(-90, 91,1), -180.01)
    right_bound = project(Re, np.arange(-90, 91,1), 180.01)
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
            
    return final_polygon.area, final_polygon


def translate(x,y):
    lat_dist = np.pi * Re / 180
    lat = (y/lat_dist)
    lon = (x/(lat_dist * np.cos(np.deg2rad(lat))))
    new_lon = remainder(lon, 360)
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
    theta = (np.pi/2.0) + elev

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


def decay_zenith_azimuth(decay_point: np.ndarray, axis: np.ndarray, decay_point_spherical: np.ndarray, axis_spherical: np.ndarray) -> np.ndarray:
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

    dot1 = np.dot(decay_point/np.linalg.norm(decay_point, axis=1)[:,None], axis)
    zenith = np.arccos(dot1)
    
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
    station: np.ndarray, decay_point: np.ndarray, geodetic: np.ndarray, decay_point_spherical: np.ndarray) -> np.ndarray:
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

    decay_vector = decay_point - station
    decay_vector = decay_vector/np.linalg.norm(decay_vector, axis=1)[:,None]

    dot1 = np.dot(decay_vector, station/np.linalg.norm(station))
    zenith = np.arccos(dot1)

    lat = np.deg2rad(geodetic[0])
    lon = np.deg2rad(geodetic[1])
    
    y = np.sin(lon - decay_point_spherical[:,2]) * np.cos(np.pi/2 - decay_point_spherical[:,1])
    x = np.cos(lat) * np.sin(np.pi/2 - decay_point_spherical[:,1]) - np.sin(lat) * np.cos(np.pi/2 - decay_point_spherical[:,1]) * np.cos(lon - decay_point_spherical[:,2])
    a = np.arctan2(y,x)
    
    azimuth = a + np.pi/2

    return zenith, azimuth


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
