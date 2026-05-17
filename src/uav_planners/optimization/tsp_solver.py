"""TSP solver for coverage path planning - optimizes cell order while preserving internal waypoint sequences."""

import math
from typing import List, Tuple
import numpy as np

from ..models.waypoint import Waypoint


class TSPSolver:
    """TSP solver for optimizing cell visitation order.
    
    This solver optimizes the order of cells (each containing a list of waypoints)
    while preserving the internal waypoint sequence within each cell.
    
    Supports multiple solving methods:
    - "ortools": Google OR-Tools for exact/optimal solutions
    - "lkh": LKH heuristic for large-scale problems
    - "nearest": Greedy nearest neighbor (fast, approximate)
    - "auto": Choose based on problem size
    
    Attributes:
        method: Solving method to use
    """
    
    def __init__(self, method: str = "auto"):
        """Initialize TSP solver.
        
        Args:
            method: Solving method ("auto", "ortools", "lkh", "nearest")
            
        Raises:
            ValueError: If method is not recognized
        """
        valid_methods = ["auto", "ortools", "lkh", "nearest"]
        if method not in valid_methods:
            raise ValueError(
                f"Unknown TSP method: {method}. "
                f"Valid options: {valid_methods}"
            )
        self.method = method
    
    def solve_cells(self, cell_waypoints: List[List[Waypoint]]) -> List[List[Waypoint]]:
        """Solve TSP to find optimal cell visitation order.
        
        Each cell contains a list of waypoints that must remain in order.
        This method determines the optimal order to visit cells, and for each
        cell determines whether to traverse forward or backward.
        
        Args:
            cell_waypoints: List of cells, each containing ordered waypoints
            
        Returns:
            Reordered list of cells (internal waypoints preserved)
        """
        if len(cell_waypoints) <= 1:
            return cell_waypoints.copy()
        
        # Select method
        method = self.method
        n_cells = len(cell_waypoints)
        if method == "auto":
            if n_cells < 50:
                method = "ortools"
            elif n_cells < 500:
                method = "lkh"
            else:
                method = "nearest"
        
        # Get cell centers for distance calculation
        cell_centers = self._get_cell_centers(cell_waypoints)
        
        if method == "ortools":
            order = self._solve_ortools(cell_centers)
        elif method == "lkh":
            order = self._solve_lkh(cell_centers)
        elif method == "nearest":
            order = self._solve_nearest_neighbor(cell_centers)
        else:
            raise ValueError(f"Unknown TSP method: {method}")
        
        # Reorder cells according to TSP solution, with direction optimization
        optimized_cells = []
        for i, cell_idx in enumerate(order):
            cell = cell_waypoints[cell_idx]
            
            # Determine direction: should we reverse this cell?
            if i > 0:
                prev_cell = optimized_cells[-1]
                prev_end = prev_cell[-1] if prev_cell else None
                curr_start = cell[0] if cell else None
                curr_end = cell[-1] if cell else None
                
                if prev_end and curr_start and curr_end:
                    # Choose direction that minimizes distance from previous cell end
                    dist_forward = self._distance(prev_end, curr_start)
                    dist_backward = self._distance(prev_end, curr_end)
                    
                    if dist_backward < dist_forward:
                        cell = cell[::-1]  # Reverse traversal
            
            optimized_cells.append(cell)
        
        return optimized_cells
    
    def _get_cell_centers(self, cell_waypoints: List[List[Waypoint]]) -> List[Tuple[float, float, float]]:
        """Get center point of each cell for distance calculation.
        
        Args:
            cell_waypoints: List of cells with waypoints
            
        Returns:
            List of (x, y, z) centers for each cell
        """
        centers = []
        for waypoints in cell_waypoints:
            if not waypoints:
                centers.append((0, 0, 0))
                continue
            
            # Calculate centroid of all waypoints in cell
            x = sum(wp.x for wp in waypoints) / len(waypoints)
            y = sum(wp.y for wp in waypoints) / len(waypoints)
            z = sum(wp.z for wp in waypoints) / len(waypoints)
            centers.append((x, y, z))
        
        return centers
    
    def _distance(self, wp1: Waypoint, wp2: Waypoint) -> float:
        """Calculate Euclidean distance between two waypoints.
        
        Args:
            wp1: First waypoint
            wp2: Second waypoint
            
        Returns:
            Euclidean distance
        """
        return math.sqrt(
            (wp1.x - wp2.x) ** 2 +
            (wp1.y - wp2.y) ** 2 +
            (wp1.z - wp2.z) ** 2
        )
    
    def _compute_distance_matrix(self, centers: List[Tuple[float, float, float]]) -> np.ndarray:
        """Compute distance matrix between cell centers.
        
        Args:
            centers: List of (x, y, z) cell centers
            
        Returns:
            NxN distance matrix
        """
        n = len(centers)
        coords = np.array(centers)
        
        # Compute Euclidean distances
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        return distances
    
    def _solve_ortools(self, centers: List[Tuple[float, float, float]]) -> List[int]:
        """Solve TSP using Google OR-Tools.
        
        Args:
            centers: List of cell centers
            
        Returns:
            Ordered indices
        """
        try:
            from ortools.constraint_solver import routing_enums_pb2
            from ortools.constraint_solver import pywrapcp
        except ImportError:
            # Fall back to nearest neighbor if OR-Tools not available
            return self._solve_nearest_neighbor(centers)
        
        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(centers)
        
        # Scale to integers (OR-Tools requires integers)
        scale = 1000  # Scale factor for precision
        distance_matrix_int = (distance_matrix * scale).astype(int)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            len(centers), 1, 0  # num_locations, num_vehicles, depot
        )
        routing = pywrapcp.RoutingModel(manager)
        
        # Define distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix_int[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(5)
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            # Extract ordered indices
            ordered = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                ordered.append(node)
                index = solution.Value(routing.NextVar(index))
            return ordered
        else:
            # Fallback if no solution found
            return list(range(len(centers)))
    
    def _solve_lkh(self, centers: List[Tuple[float, float, float]]) -> List[int]:
        """Solve TSP using LKH heuristic.
        
        Args:
            centers: List of cell centers
            
        Returns:
            Ordered indices
        """
        try:
            from .solvers.lkh_wrapper import LKHWrapper
        except ImportError:
            return self._solve_nearest_neighbor(centers)
        
        lkh = LKHWrapper()
        
        if not lkh.is_available:
            return self._solve_nearest_neighbor(centers)
        
        # Convert centers to waypoints for LKH
        temp_waypoints = [
            Waypoint(x=cx, y=cy, z=cz, heading_deg=0, gimbal_pitch_deg=-90)
            for cx, cy, cz in centers
        ]
        
        result = lkh.solve(temp_waypoints)
        
        if result is None:
            return self._solve_nearest_neighbor(centers)
        
        # Map back to indices
        index_map = {id(wp): i for i, wp in enumerate(temp_waypoints)}
        return [index_map[id(wp)] for wp in result]
    
    def _solve_nearest_neighbor(self, centers: List[Tuple[float, float, float]]) -> List[int]:
        """Solve TSP using greedy nearest neighbor algorithm.
        
        Args:
            centers: List of cell centers
            
        Returns:
            Ordered indices (approximate solution)
        """
        if not centers:
            return []
        
        n = len(centers)
        distance_matrix = self._compute_distance_matrix(centers)
        
        # Start from first center
        unvisited = set(range(1, n))
        current = 0
        order = [current]
        
        while unvisited:
            # Find nearest unvisited center
            nearest = min(unvisited, key=lambda i: distance_matrix[current][i])
            order.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return order
    
    # Legacy method for backward compatibility
    def solve(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Legacy method for single list of waypoints.
        
        Deprecated: Use solve_cells() for cell-based optimization.
        
        Args:
            waypoints: Unordered list of waypoints
            
        Returns:
            Ordered list of waypoints
        """
        import warnings
        warnings.warn(
            "solve() is deprecated. Use solve_cells() for cell-based optimization.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Convert to cell format (single cell)
        return self.solve_cells([waypoints])[0]