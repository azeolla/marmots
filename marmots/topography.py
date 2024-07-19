from typing import Any, NamedTuple, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
import triangle as tr
from tqdm import tqdm

from marmots.constants import Re

from numba import jit, njit

import rasterio
from rasterio.transform import from_origin

import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

latlon2xyz = Transformer.from_crs("EPSG:4326", "EPSG:4328")
xyz2latlon = Transformer.from_crs("EPSG:4328", "EPSG:4326")
sinusoidal = Transformer.from_crs("EPSG:4326", "ESRI:54008")
inverse_sinusoidal = Transformer.from_crs("ESRI:54008", "EPSG:4326")

# create a named tuple to store our geometry information
TotalArea = NamedTuple(
    "TotalArea",
    [
        ("triangles", np.ndarray),
        ("centroids", np.ndarray),
        ("center", np.ndarray),
        ("stations_geocentric", np.ndarray),
        ("stations_geodetic", np.ndarray),
        ("orientations", np.ndarray),
        ("fov", np.ndarray)
    ],
)


def to_geocentric(lat, lon, height):
    xyz = np.array(latlon2xyz.transform(lat, lon, height)).T
    return xyz

def to_geodetic(points):
    latlon = np.array(xyz2latlon.transform(points[:,0], points[:,1], points[:,2])).T
    return latlon

def geocentric2local(points, origin):
    
    new = points - origin
    
    llz = to_geodetic(np.array([origin])*1e3)[0]

    R1 =  rotation_matrix(np.array([1,0,0]), -np.deg2rad(90-llz[0]))
    R2 =  rotation_matrix(np.array([0,0,1]), -np.deg2rad(90+llz[1]))
    
    R3 = R1@R2

    v = new.reshape(-1, 3).T
    enu = (R3 @ v).T.reshape(new.shape)
    
    return enu

def project(points):
    x, y = sinusoidal.transform(points[:,0], points[:,1])
    return x, y

def unproject(x, y):
    lat, lon = inverse_sinusoidal.transform(x, y)
    return lat, lon

def rotation_matrix(axis, theta):
    R = np.empty((3,3))
    R[0] = np.array([np.cos(theta)+axis[0]**2 * (1-np.cos(theta)), axis[0]*axis[1]*(1-np.cos(theta)) - axis[2]*np.sin(theta),axis[0]*axis[2]*(1-np.cos(theta)) + axis[1]*np.sin(theta)])
    R[1] = np.array([axis[1]*axis[0]*(1-np.cos(theta)) + axis[2]*np.sin(theta),np.cos(theta)+axis[1]**2 * (1-np.cos(theta)),axis[1]*axis[2]*(1-np.cos(theta)) - axis[0]*np.sin(theta)])
    R[2] = np.array([axis[2]*axis[0]*(1-np.cos(theta)) - axis[1]*np.sin(theta),axis[2]*axis[1]*(1-np.cos(theta)) + axis[0]*np.sin(theta),np.cos(theta)+axis[2]**2 * (1-np.cos(theta))])
    
    return R 

def lat_lon_to_pixel(latitudes, longitudes, dataset):
    """
    Convert latitude and longitude arrays to pixel coordinates.
    """
    # Get the transformation function
    x, y = dataset.bounds.left, dataset.bounds.top
    transform = from_origin(x, y, dataset.res[0], dataset.res[1])
    
    # Convert latitude and longitude arrays to pixel coordinates
    cols, rows = ~transform * (longitudes, latitudes)
    
    return rows.astype(int), cols.astype(int)


def get_elevation_at_lat_lon(lat, lon, dataset):
    """
    Get elevation at a given latitude and longitude.
    """
    # Convert latitude and longitude to pixel coordinates
    rows, cols = lat_lon_to_pixel(lat, lon, dataset)
    
    # Read elevation value
    elevation = dataset.read(1)[rows,cols]
    
    return elevation


def get_elevation(latitudes, longitudes):
    elev = np.empty(latitudes.size)
    floored = np.array([np.floor(latitudes), np.floor(longitudes)]).T.astype(int)
    latlon = np.unique(floored, axis=0)
    
    for i in range(latlon.shape[0]):
        if latlon[i][0] >= 0:
            lat = 'N' + str(abs(latlon[i][0]))
        else:
            lat = 'S' + str(abs(latlon[i][0]))
        if latlon[i][1] >= 0:
            lon = 'E' + str(abs(latlon[i][1])).zfill(3)
        else:
            lon = 'W' + str(abs(latlon[i][1])).zfill(3)
            
        file_path = f"/data2/beacon/srtm30m/{lat}{lon}.hgt"
        dataset = rasterio.open(file_path)
        
        idx = (floored[:,0] == latlon[i][0]) & (floored[:,1] == latlon[i][1])
        rows, cols = lat_lon_to_pixel(latitudes[idx], longitudes[idx], dataset)
        elev[idx] = get_elevation_at_lat_lon(latitudes[idx], longitudes[idx], dataset)
        dataset.close()
        
    return elev

def rotate_around_axis(vector, axis, theta):
    R = np.empty((3,3))
    R[0] = np.array([np.cos(theta)+axis[0]**2 * (1-np.cos(theta)), axis[0]*axis[1]*(1-np.cos(theta)) - axis[2]*np.sin(theta),axis[0]*axis[2]*(1-np.cos(theta)) + axis[1]*np.sin(theta)])
    R[1] = np.array([axis[1]*axis[0]*(1-np.cos(theta)) + axis[2]*np.sin(theta),np.cos(theta)+axis[1]**2 * (1-np.cos(theta)),axis[1]*axis[2]*(1-np.cos(theta)) - axis[0]*np.sin(theta)])
    R[2] = np.array([axis[2]*axis[0]*(1-np.cos(theta)) - axis[1]*np.sin(theta),axis[2]*axis[1]*(1-np.cos(theta)) + axis[0]*np.sin(theta),np.cos(theta)+axis[2]**2 * (1-np.cos(theta))])
    
    return R @ vector

@njit
def normalize(vec: np.ndarray):
    # normalize along axis=1
    norm = np.sqrt(vec[:,0]**2 +vec[:,1]**2 + vec[:,2]**2)
    return vec/np.expand_dims(norm, 1)

def visibile_horizon_mesh(lat, lon, height, orientations, fov, distance_beyond=200):

    stations_geocentric = to_geocentric(lat, lon, height*1e3)
    stations_geodetic = np.array([lat, lon, height*1e3]).T

    polygons = []
    for i in tqdm(range(stations_geocentric.shape[0])):
        stat = stations_geocentric[i]/np.linalg.norm(stations_geocentric[i])
        east = -np.cross(stat, np.array([0,0,1]))
        east /= np.linalg.norm(east)
        orientation = rotate_around_axis(east, stat, np.deg2rad(orientations[i]))
        orientation /= np.linalg.norm(orientation)
        angles = np.arange(-fov[i]/2,fov[i]/2+1,0.25)
        vectors = np.zeros((angles.size,3))
        # A fan of vectors spanning the field of view
        for j in range(angles.shape[0]):
            vectors[j] = rotate_around_axis(orientation, stat, np.deg2rad(angles[j]))

        # Find the elevation at 1 km intervals along each vector
        dist = np.arange(1,375,1)*1e3
        llz = np.array([to_geodetic((stations_geocentric[i] + dist[:,None]*vec)) for vec in normalize(vectors)])
        z = get_elevation(np.concatenate(llz)[:,0], np.concatenate(llz)[:,1]).reshape(llz.shape[0:2])

        points = np.zeros((angles.size + 3,2))
        points[0] = stations_geodetic[i][0:2] # the location of the station

        surface = np.array([llz[0][:,0], llz[0][:,1], z[0]]).T
        xyz = to_geocentric(surface[:,0], surface[:,1], surface[:,2]) 
        local = geocentric2local(xyz, stations_geocentric[i])
        R = np.sqrt(local[:,0]**2 + local[:,1]**2)

        # the visible horizon is the max of xyz/R
        # we extend some distance beyond the visible horizon to allow taus to pass through terrain
        idx = np.clip(np.argmax(local[:,2]/R)+distance_beyond, 0, dist.size-1)
        points[1] = surface[int(idx/2)][0:2] # half way to the horizon at the first edge
        points[2] = surface[int(idx)][0:2] # the horizon at the first edge
        # do this for each vector
        for k in range(1,vectors.shape[0]):
            surface = np.array([llz[k][:,0], llz[k][:,1], z[k]]).T
            xyz = to_geocentric(surface[:,0], surface[:,1], surface[:,2]) 
            local = geocentric2local(xyz, stations_geocentric[i])
            R = np.sqrt(local[:,0]**2 + local[:,1]**2)
            idx = np.clip(np.argmax(local[:,2]/R)+distance_beyond, 0, dist.size-1)
            points[k+2] = surface[int(idx)][0:2]
        points[-1] = surface[int(idx/2)][0:2] # half way to horizon at the second edge

        # identify areas that cross on the antimeridian. These points need to be moved all to the same side
        s = np.sum(points[:,1] > 0)
        wrapped_over_antimeridian = (s != 0) & (s != points[:,1].size) & (np.mean(abs(points[:,1])) > 90)
        if wrapped_over_antimeridian:
            points[:,1][points[:,1] < 0] += 360

        # sinusoidal projection allows us to work in 2D, while conserving the shape's area
        x,y = project(points)

        pp = list(zip(x/1e3,y/1e3))

        polygons.append(Polygon(pp).buffer(0))
        
    # the union of all the areas
    multi = unary_union(polygons)

    # triangulate the union area in order to randomly sample points uniformly
    triangles = []
    if type(multi) == shapely.geometry.multipolygon.MultiPolygon:
        for i in range(len(multi.geoms)):
            vertices = np.array(multi.geoms[i].exterior.coords)[:-1]

            start = np.arange(0, vertices.shape[0])
            end = np.roll(start,-1)
            segments = np.column_stack((start,end))

            shape = dict(vertices = vertices, segments = segments)

            tri = tr.triangulate(shape, 'pa0.5')

            triangles.append(tri['vertices'][tri['triangles']])
    else:
        vertices = np.array(multi.exterior.coords)[:-1]

        start = np.arange(0, vertices.shape[0])
        end = np.roll(start,-1)
        segments = np.column_stack((start,end))

        shape = dict(vertices = vertices, segments = segments)

        tri = tr.triangulate(shape, 'pa0.5')

        triangles.append(tri['vertices'][tri['triangles']])

    triangles = np.concatenate(triangles)

    latitude, longitude = unproject(np.concatenate(triangles)[:,0]*1e3, np.concatenate(triangles)[:,1]*1e3)

    z = get_elevation(latitude, longitude)

    cartesian = to_geocentric(latitude, longitude, z).reshape(triangles.shape[0],3,3)/1e3

    centroids = np.mean(cartesian,axis=1)

    return TotalArea(cartesian, centroids, np.mean(centroids,axis=0), stations_geocentric/1e3, stations_geodetic, orientations, fov)