"""
Evaluate properties of tau decays.
"""
import numpy as np
import marmots.mesh as mesh



def line_of_sight(decay_points, station, TotalArea, BVH):
    
    origin = TotalArea.center
    
    local_station = mesh.geocentric2local(station, origin)
    local_points = mesh.geocentric2local(decay_points, origin)
    
    ints = []
    for i in range(decay_points.shape[0]):
        local_ints = mesh.intersect_bvh_with_segment(BVH, local_station, local_points[i])
        ints.append(local_ints)
    return ~np.array(ints).astype(bool)

def probability(decay_length: np.ndarray, dbeacon: np.ndarray) -> np.ndarray:
    """
    Return the probability that a tau with decay length `decay_length` decays
    before traveling `dbeacon` (km)`

    Parameter
    ---------
    decay_length: np.ndarray
        The decay length of the tau (in km).
    dbeacon: np.ndarray
        The distance from exit to BEACON (in km).

    Returns
    -------
    Pdecay: np.ndarray
        The decay probability at each tau energy.
    """
    return 1.0 - np.exp(-dbeacon / decay_length)

