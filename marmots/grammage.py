import marmots.mesh as mesh
from scipy.interpolate import interp1d
import numpy as np

from marmots import data_directory

# load the PREM model
PREM = np.loadtxt(data_directory + "/PREM_1s.csv", delimiter=',')
radius = PREM[:,0]
density = PREM[:,2]*1e15
density[:4] = density[5] # set water densities to be bedrock

# Initialize PREM interpolator
f = interp1d(radius, density, bounds_error = False, fill_value=(density[-1], density[0]))

# Load grammage as a function of emergence angle
data = np.loadtxt(data_directory + "/tauexit/grammage.dat", skiprows=1, delimiter=',')
angle = data[:,0]
g = data[:,1]
# Initialize grammage interpolator
f2 = interp1d(g, 180-angle, bounds_error = False, fill_value='extrapolate')

def entrance_point(vector, exit_point):
    """
    Given a vector, find where it first enters a sphere.
    
    #https://diegoinacio.github.io/computer-vision-notebooks-page/pages/ray-intersection_sphere.html
    
    #note: this assumes that the vector is pointed toward the source
    """
    radius = np.linalg.norm(exit_point)
    
    t = np.dot(exit_point, vector)
    p = exit_point - vector*t
    d = np.linalg.norm(p)
    
    i = np.sqrt(np.maximum(radius**2 - d**2, 0))

    P = exit_point - vector*(t - i)
        
    return P


def grammage(exitpoint, ints, axis):
    """
    Given an exit-point, a list of mesh intersections, and the particle axis, calculates
    the amount of grammage traversed.
    """

    # if mesh intersections exist:
    if ints.size > 0:
        distances = np.linalg.norm(exitpoint - ints, axis=1)
        in_order = np.argsort(distances)
        # make sure intersection points are in order
        distances = distances[in_order]

        ground=True
        traversed = 0
        # loop over intersection points, alternating between ground and air
        for i in range(distances.size):
            if ground:
                if i == 0:
                    traversed += distances[i]
                else:
                    traversed += distances[i] - distances[i-1]
                ground = False
            else:
                ground = True

        # if the last intersection point is air, then output the grammage
        if ground == False:
            return traversed * 2.6e+15 * 1e-10
        # if the last intersection point is in ground, then we need to continue propagating beyond the mesh
        # for this, we assume the Earth is a sphere
        else: 
            # find where the vector entered a sphere with radius = R_e
            entrance = entrance_point(axis, ints[in_order][-1])
            distance = np.linalg.norm(entrance - ints[in_order][-1])
            # sample points along this chord
            points = ints[in_order][-1] + np.linspace(0, distance, 101)[:,None]*axis
            radii = np.linalg.norm(points, axis=1)
            # calculate the grammage traversed
            return (traversed * 2.6e+15 * 1e-10) + np.trapz(f(radii), dx=distance/100)*1e-10

    # if mesh intersections == 0, then the tau is steeply up-going
    # for this, we assume the Earth is a sphere
    else: 
        # find where the vector entered a sphere with radius = R_e
        entrance = entrance_point(axis, exitpoint)
        distance = np.linalg.norm(entrance - exitpoint)
        # sample points along this chord
        points = exitpoint + np.linspace(0, distance, 101)[:,None]*axis
        radii = np.linalg.norm(points, axis=1)
        # calculate the grammage traversed
        return np.trapz(f(radii), dx=distance/100)*1e-10
    

def find_grammage(trials, axis, TotalArea, BVH):
    """
    Given a list of exit points (trials), the particle axis, the triangulated surface mesh, and the bounding volume hierachy, 
    intersections with the mesh in the backwards direction are found, and the amount of grammage traversed for each tau event
    is then calculated.
    """
    
    gram = np.zeros(trials.shape[0])

    # transform to local coordinates
    # this improves the speed of intersection calculation
    origin = TotalArea.center
    local_trials = mesh.geocentric2local(trials, origin)
    
    local_axis = mesh.geocentric2local(origin+axis, origin)
    local_axis /= np.linalg.norm(local_axis)

    # loop over exit points
    for i in range(trials.shape[0]):
        # find intersections between the particle axis and the triangulated surface mesh
        local_ints = np.array(mesh.intersect_bvh(BVH, local_trials[i], -local_axis))
        if local_ints.size > 0:
            # transform to geocentric coordinates
            ints = mesh.local2geocentric(local_ints, origin)
        else:
            ints = np.array([])
        # calculate grammage
        gram[i] = grammage(trials[i], ints, -axis)
    
    return gram

def find_exit_angle(trials, axis, TotalArea, BVH):
    """
    The P_exit and tau energy Look-Up Tables are functions of exit angle, assuming a spherical Earth.
    Here, we've calculated grammage. To get around this, we "convert" grammage to exit angle, given the exit angles
    and grammages used in the NuTauSim simulations.
    """
    
    grammage = find_grammage(trials, axis, TotalArea, BVH)
    
    exit_angle = f2(grammage)
    
    return exit_angle

