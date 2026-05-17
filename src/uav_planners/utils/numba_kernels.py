"""Numba JIT-compiled kernels for geometric distance and collision checks."""

import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True, fastmath=True)
def fast_distance_squared(p1, p2):
    """Fast squared distance (no sqrt)."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return dx*dx + dy*dy + dz*dz


@jit(nopython=True, cache=True, fastmath=True)
def fast_distance(p1, p2):
    """Fast Euclidean distance."""
    return np.sqrt(fast_distance_squared(p1, p2))


@jit(nopython=True, cache=True, fastmath=True)
def batch_collision_check_single(point, obstacle_points, safety_margin_sq):
    """Check single point against all obstacles."""
    for j in range(len(obstacle_points)):
        dx = point[0] - obstacle_points[j, 0]
        dy = point[1] - obstacle_points[j, 1]
        dz = point[2] - obstacle_points[j, 2]
        dist_sq = dx*dx + dy*dy + dz*dz
        if dist_sq < safety_margin_sq:
            return True  # Collision
    return False  # Safe


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def batch_collision_check(points, obstacle_points, safety_margin):
    """Check multiple points for collision in parallel.
    
    Args:
        points: Array of points to check [N, 3]
        obstacle_points: Obstacle point cloud [M, 3]
        safety_margin: Minimum safe distance
        
    Returns:
        True if all points are collision-free
    """
    n_points = len(points)
    margin_sq = safety_margin * safety_margin
    
    # Array to store collision results per point
    collision_found = False
    
    for i in prange(n_points):
        if collision_found:
            continue  # Early exit if collision already found
        
        point = points[i]
        
        # Check against all obstacles
        for j in range(len(obstacle_points)):
            dx = point[0] - obstacle_points[j, 0]
            dy = point[1] - obstacle_points[j, 1]
            dz = point[2] - obstacle_points[j, 2]
            dist_sq = dx*dx + dy*dy + dz*dz
            
            if dist_sq < margin_sq:
                collision_found = True
                break
    
    return not collision_found


@jit(nopython=True, cache=True, fastmath=True)
def interpolate_edge(start, end, num_samples):
    """Interpolate points along edge."""
    points = np.empty((num_samples, 3))
    for i in range(num_samples):
        t = i / (num_samples - 1)
        points[i, 0] = start[0] + t * (end[0] - start[0])
        points[i, 1] = start[1] + t * (end[1] - start[1])
        points[i, 2] = start[2] + t * (end[2] - start[2])
    return points


@jit(nopython=True, cache=True, fastmath=True)
def nearest_node_idx(query_point, nodes_array):
    """Find index of nearest node using Numba.
    
    Args:
        query_point: [x, y, z]
        nodes_array: [N, 3] array of node positions
        
    Returns:
        Index of nearest node
    """
    min_dist_sq = 1e10
    min_idx = 0
    
    for i in range(len(nodes_array)):
        dx = query_point[0] - nodes_array[i, 0]
        dy = query_point[1] - nodes_array[i, 1]
        dz = query_point[2] - nodes_array[i, 2]
        dist_sq = dx*dx + dy*dy + dz*dz
        
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            min_idx = i
    
    return min_idx


@jit(nopython=True, cache=True, fastmath=True)
def path_cost(path_points):
    """Calculate total path cost (sum of edge lengths)."""
    total = 0.0
    for i in range(len(path_points) - 1):
        total += fast_distance(path_points[i], path_points[i + 1])
    return total
