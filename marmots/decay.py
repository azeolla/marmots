"""
Evaluate properties of tau decays.
"""
import numpy as np
import marmots.mesh as mesh



def line_of_sight(decay_points, station, TotalArea, BVH):
    """
    Determines whether or not the line-of-sight between a station and a 
    decay point is interrupted. 
    """
    
    origin = TotalArea.center
    
    local_station = mesh.geocentric2local(station, origin)
    local_points = mesh.geocentric2local(decay_points, origin)
    
    ints = []
    for i in range(decay_points.shape[0]):
        local_ints = mesh.intersect_bvh_with_segment(BVH, local_station, local_points[i])
        ints.append(local_ints)
    return ~np.array(ints).astype(bool)

def probability(decay_length: np.ndarray, dobservatory: np.ndarray) -> np.ndarray:
    """
    Return the probability that a tau with decay length `decay_length` decays
    before traveling `dobservatory` (km)`

    Parameter
    ---------
    decay_length: np.ndarray
        The decay length of the tau (in km).
    dobservatory: np.ndarray
        The distance from exit to observatory(in km).

    Returns
    -------
    Pdecay: np.ndarray
        The decay probability at each tau energy.
    """
    return 1.0 - np.exp(-dobservatory / decay_length)

def tau_path_clear(exit_points, decay_points, TotalArea, BVH):
    """
    Determines whether the trajectory from each exit point to its
    corresponding decay point is interrupted by terrain.
    """
    origin = TotalArea.center
    local_exit   = mesh.geocentric2local(exit_points, origin)
    local_decays = mesh.geocentric2local(decay_points, origin)

    ints = []
    for i in range(exit_points.shape[0]):
        local_ints = mesh.intersect_bvh_with_segment(
            BVH, local_exit[i], local_decays[i]
        )
        ints.append(local_ints)

    return ~np.array(ints).astype(bool)