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