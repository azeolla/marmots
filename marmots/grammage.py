import marmots.mesh as mesh
from scipy.interpolate import interp1d
import numpy as np


PREM = np.loadtxt("/home/avz5228/Downloads/PREM_1s.csv", delimiter=',')
radius = PREM[:,0]
density = PREM[:,2]*1e15
density[:4] = density[5] # set water densities to be bedrock

f = interp1d(radius, density, bounds_error = False, fill_value=(density[-1], density[0]))

data = np.loadtxt("/home/avz5228/marmots/data/tauexit/grammage.dat", skiprows=1, delimiter=',')
angle = data[:,0]
g = data[:,1]
f2 = interp1d(g, 180-angle, bounds_error = False, fill_value='extrapolate')

def entrance_point(vector, exit_point):
    
    #https://diegoinacio.github.io/computer-vision-notebooks-page/pages/ray-intersection_sphere.html
    
    #note: this assumes that the vector is pointed toward the source
    radius = np.linalg.norm(exit_point)
    
    t = np.dot(exit_point, vector)
    p = exit_point - vector*t
    d = np.linalg.norm(p)
    
    i = np.sqrt(radius**2 - d**2)

    P = exit_point - vector*(t - i)
        
    return P


def grammage(exitpoint, ints, axis):
    
    if ints.size > 0:
        distances = np.linalg.norm(exitpoint - ints, axis=1)
        in_order = np.argsort(distances)
        distances = distances[in_order]

        ground=True
        traversed = 0
        for i in range(distances.size):
            if ground:
                if i == 0:
                    traversed += distances[i]
                else:
                    traversed += distances[i] - distances[i-1]
                ground = False
            else:
                ground = True

        if ground == False:
            return traversed * 2.6e+15 * 1e-10

        else: 
            entrance = entrance_point(axis, ints[in_order][-1])
            distance = np.linalg.norm(entrance - ints[in_order][-1])
            points = ints[in_order][-1] + np.linspace(0, distance, 101)[:,None]*axis
            radii = np.linalg.norm(points, axis=1)
            return (traversed * 2.6e+15 * 1e-10) + np.trapz(f(radii), dx=distance/100)*1e-10
        
    else: 
        entrance = entrance_point(axis, exitpoint)
        distance = np.linalg.norm(entrance - exitpoint)
        points = exitpoint + np.linspace(0, distance, 101)[:,None]*axis
        radii = np.linalg.norm(points, axis=1)
        return np.trapz(f(radii), dx=distance/100)*1e-10
    

def find_grammage(trials, axis, TotalArea, BVH):
    
    gram = np.zeros(trials.shape[0])
    
    origin = TotalArea.center
    local_trials = mesh.geocentric2local(trials, origin)
    
    local_axis = mesh.geocentric2local(origin+axis, origin)
    local_axis /= np.linalg.norm(local_axis)
    
    for i in range(trials.shape[0]):
        local_ints = np.array(mesh.intersect_bvh(BVH, local_trials[i], -local_axis))
        if local_ints.size > 0:
            ints = mesh.local2geocentric(local_ints, origin)
        else:
            ints = np.array([])
        gram[i] = grammage(trials[i], ints, -axis)
    
    return gram

def find_exit_angle(trials, axis, TotalArea, BVH):
    
    grammage = find_grammage(trials, axis, TotalArea, BVH)
    
    exit_angle = f2(grammage)
    
    return exit_angle

