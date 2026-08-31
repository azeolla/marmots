import numpy as np
from numba import njit
from typing import NamedTuple
from marmots.topography import to_geodetic


class BVHNode:
    """
    This class generates a Bounding Volume Hierarchy.
    Calculating intersections is super, super slow without doing this.
    Essentially, this organizes all the triangles into smaller and smaller boxes.
    """
    def __init__(self, bounding_box, left=None, right=None, triangles=None):
        self.bounding_box = bounding_box
        self.left = left
        self.right = right
        self.triangles = triangles

def compute_bounding_box(triangle):
    # Compute the bounding box of a triangle
    min_coords = np.min(triangle, axis=0)
    max_coords = np.max(triangle, axis=0)
    return min_coords, max_coords

def build_bvh_tree(triangles, max_triangles_per_node=100):
    # Recursively build the BVH tree
    
    triangle_boxes = np.array([compute_bounding_box(triangle) for triangle in triangles])
    
    def build_node(triangle_indices):
        if len(triangle_indices) <= max_triangles_per_node:
            # Leaf node
            node_triangles = triangles[triangle_indices]
            node_boxes = np.concatenate(triangle_boxes[triangle_indices])
            return BVHNode(np.array(compute_bounding_box(node_boxes)), triangles=node_triangles)
        else:
            # Interior node
            # Choose a splitting axis (e.g., axis with the largest range)
            node_boxes = np.concatenate(triangle_boxes[triangle_indices])
            axis = np.argmax(np.ptp(node_boxes, axis=0)) 
            
            # Choose a splitting point (e.g., median along the chosen axis)
            split_value = np.median(triangle_boxes[triangle_indices][:, :, axis])
            
            left_indices = np.where(triangle_boxes[triangle_indices][:, 0, axis] <= split_value)[0]
            right_indices = np.where(triangle_boxes[triangle_indices][:, 0, axis] > split_value)[0]

            left_node = build_node(triangle_indices[left_indices])
            right_node = build_node(triangle_indices[right_indices])
            return BVHNode(np.array(compute_bounding_box(node_boxes)), left=left_node, right=right_node)

    # Start building the BVH tree
    root_node = build_node(np.arange(triangles.shape[0]))
    return root_node


@njit
def intersect_ray_aabb(ray_origin, ray_direction, bounding_box):
    """
    Determines if a ray intersects a bounding box. Used in the BVH tree intersection algorithm.
    """
    
    t_min = np.divide(bounding_box[0] - ray_origin, ray_direction)
    t_max = np.divide(bounding_box[1] - ray_origin, ray_direction)
    
    t_enter = np.max(np.minimum(t_min, t_max))
    t_exit = np.min(np.maximum(t_min, t_max))
    
    # Check if the ray is inside the bounding box
    #inside_bbox = np.all(ray_origin >= bounding_box[0]) and np.all(ray_origin <= bounding_box[1])
    
    return (t_exit >= 0) and (t_enter <= t_exit)


def intersect_bvh(root_node, ray_origin, ray_direction):
    """
    Uses the Bounding Volume Hierarchy to find intersections. First it checks for intersections
    with each bounding box. At the end of the tree, it then checks for intersections with triangles.
    Outputs the intersection locations. Used in the grammage calculation.
    """
    
    stack = [root_node]
    intersections = []

    while stack:
        node = stack.pop()

        if intersect_ray_aabb(ray_origin, ray_direction, node.bounding_box):
            if node.triangles is not None:
                
                # Leaf node, check intersections with triangles
                ints = triangle_intersections(ray_origin, ray_direction, node.triangles)
                intersections.extend(ints)
            else:
                # Interior node, continue traversal
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)
    # return the location of all intersections
    return intersections


@njit(fastmath=True)
def triangle_intersections(origin, axis, triangles):
    """
    Fast algorithm for calculating the locations at which a vector (axis) extending 
    from an origin intersects a mesh of triangles.
    """

    e1 = triangles[:,1]-triangles[:,0]
    e2 = triangles[:,2]-triangles[:,0]
    N = np.cross(e1,e2)
    det = -np.dot(N, axis)
    invdet = 1.0/det
    AO  = origin - triangles[:,0]
    DAO = np.cross(AO, axis)
    u =  np.sum(e2*DAO, axis=1) * invdet
    v = -np.sum(e1*DAO, axis=1) * invdet
    t =  np.sum(AO*N, axis=1)  * invdet

    t1 = np.abs(det) >= 1e-6
    t2 = t >= 1e-6
    t3 = u >= 0.0
    t4 = v >= 0.0
    t5 = (u+v) <= 1.0

    mask = t1 & t2 & t3 & t4 & t5

    return origin + t.reshape(-1,1)[mask]*axis


@njit
def intersect_segment_aabb(segment_start, segment_end, bounding_box):
    """
    Determines if a line segment intersects a bounding box.
    """
    segment_direction = segment_end - segment_start
    inv_dir = 1.0 / segment_direction

    t_min = np.multiply(bounding_box[0] - segment_start, inv_dir)
    t_max = np.multiply(bounding_box[1] - segment_start, inv_dir)

    t_enter = np.max(np.minimum(t_min, t_max))
    t_exit = np.min(np.maximum(t_min, t_max))
    
    # Check if the intersection is within the segment's length
    return t_enter <= t_exit and t_exit >= 0.0 and t_enter <= 1.0


def intersect_bvh_with_segment(root_node, segment_start, segment_end):
    """
    Uses the Bounding Volume Hierarchy to find intersections. In this case, we use line segments
    and only care about whether or not intersections occur. Used in the line-of-sight calculation.
    """
    stack = [root_node]

    while stack:
        node = stack.pop()

        if intersect_segment_aabb(segment_start, segment_end, node.bounding_box):
            if node.triangles is not None:
                # Leaf node, check intersections with triangles
                if triangle_segment_intersections(segment_start, segment_end, node.triangles):
                    return True
            else:
                # Interior node, continue traversal
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)

    return False


@njit(fastmath=True)
def triangle_segment_intersections(segment_start, segment_end, triangles):
    """
    Determines which triangles a line segment intersects. Outputs a list of bool values.
    """
    segment_direction = segment_end - segment_start
    segment_length = np.linalg.norm(segment_direction)
    segment_direction /= segment_length

    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    N = np.cross(e1, e2)
    det = -np.dot(N, segment_direction)
    invdet = 1.0 / det
    AO = segment_start - triangles[:, 0]
    DAO = np.cross(AO, segment_direction)
    u = np.sum(e2 * DAO, axis=1) * invdet
    v = -np.sum(e1 * DAO, axis=1) * invdet
    t = np.sum(AO * N, axis=1) * invdet

    t1 = np.abs(det) >= 1e-6
    # this ensures that we don't count intersections very close to the station.
    # intersections near the station can be misleading, depending on the mesh resolution.
    t2 = t > 0.5 
    t3 = u >= 0.0
    t4 = v >= 0.0
    t5 = (u + v) <= 1.0
    t6 = t < segment_length 

    mask = t1 & t2 & t3 & t4 & t5 & t6

    return np.any(mask)


def rotation_matrix(axis, theta):
    """
    Generates a rotation matrix which rotates a vector by angle theta around axis.
    """
    R = np.empty((3,3))
    R[0] = np.array([np.cos(theta)+axis[0]**2 * (1-np.cos(theta)), axis[0]*axis[1]*(1-np.cos(theta)) - axis[2]*np.sin(theta),axis[0]*axis[2]*(1-np.cos(theta)) + axis[1]*np.sin(theta)])
    R[1] = np.array([axis[1]*axis[0]*(1-np.cos(theta)) + axis[2]*np.sin(theta),np.cos(theta)+axis[1]**2 * (1-np.cos(theta)),axis[1]*axis[2]*(1-np.cos(theta)) - axis[0]*np.sin(theta)])
    R[2] = np.array([axis[2]*axis[0]*(1-np.cos(theta)) - axis[1]*np.sin(theta),axis[2]*axis[1]*(1-np.cos(theta)) + axis[0]*np.sin(theta),np.cos(theta)+axis[2]**2 * (1-np.cos(theta))])
    
    return R 


def geocentric2local(points, origin):
    """
    Converts geocentric coordinates to local ENU coordinates.
    """
    
    new = points - origin
    
    llz = to_geodetic(np.array([origin])*1e3)[0]

    R1 =  rotation_matrix(np.array([1,0,0]), -np.deg2rad(90-llz[0]))
    R2 =  rotation_matrix(np.array([0,0,1]), -np.deg2rad(90+llz[1]))
    
    R3 = R1@R2

    v = new.reshape(-1, 3).T
    enu = (R3 @ v).T.reshape(new.shape)
    
    return enu


def local2geocentric(points, origin):
    """
    Converts local ENU coordinates to geocentric coordinates.
    """
    
    llz = to_geodetic(np.array([origin])*1e3)[0]

    R1 =  rotation_matrix(np.array([1,0,0]), np.deg2rad(90-llz[0]))
    R2 =  rotation_matrix(np.array([0,0,1]), np.deg2rad(90+llz[1]))
    
    R3 = R2@R1

    v = points.reshape(-1, 3).T
    geocentric = origin + (R3 @ v).T.reshape(points.shape)
    
    return geocentric


def create_BVH_tree(mesh: NamedTuple):
    
    origin = mesh.center
    local_surface = geocentric2local(np.concatenate(mesh.triangles), origin).reshape(mesh.triangles.shape)
    local = build_bvh_tree(local_surface)
    
    return local
    
    