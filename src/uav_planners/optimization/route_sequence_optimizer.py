"""Route sequence optimizer for TSP-based route ordering.

This module provides optimization of route execution order for multi-route
UAV coverage planning (e.g., oblique photography with nadir, north, east, south, west).
"""

import math
from typing import List, Optional, Tuple
import numpy as np
import logging

from ..models.route_collection import RouteCollection, RouteMetadata
from ..models.waypoint import Waypoint


logger = logging.getLogger(__name__)


class RouteSequenceOptimizer:
    """Optimizer for determining optimal route execution order.
    
    Uses TSP (Traveling Salesman Problem) to find the optimal sequence
    for executing multiple routes, minimizing transition distances between
    route exit and entry points.
    
    This is particularly useful for oblique photography where multiple
    routes (nadir, north, east, south, west) need to be flown in an
    order that minimizes total flight time.
    
    Attributes:
        method: TSP solving method ("greedy", "ortools", "auto")
    
    Example:
        >>> optimizer = RouteSequenceOptimizer(method="greedy")
        >>> optimized_collection = optimizer.optimize(route_collection)
    """
    
    def __init__(self, method: str = "greedy"):
        """Initialize the route sequence optimizer.
        
        Args:
            method: TSP solving method. Options:
                - "greedy": Greedy nearest neighbor (fast, default)
                - "ortools": Google OR-Tools for near-optimal solutions
                - "auto": Choose based on problem size
                
        Raises:
            ValueError: If method is not recognized
        """
        valid_methods = ["greedy", "ortools", "auto"]
        if method not in valid_methods:
            raise ValueError(
                f"Unknown TSP method: {method}. "
                f"Valid options: {valid_methods}"
            )
        self.method = method
    
    def optimize(self, collection: RouteCollection) -> RouteCollection:
        """Optimize the execution order of routes in a collection.
        
        Builds a distance matrix from route exit points to entry points,
        then solves TSP to find the optimal execution order.
        
        Args:
            collection: RouteCollection containing routes to optimize
            
        Returns:
            New RouteCollection with routes in optimized order.
            Returns original collection if single route or empty.
        """
        if collection.route_count <= 1:
            logger.debug("Single or empty route collection, no optimization needed")
            return collection
        
        logger.info(f"Optimizing sequence for {collection.route_count} routes")
        
        # Extract entry and exit points for each route
        entry_points, exit_points = self._extract_route_endpoints(collection)
        
        # Build distance matrix (exit point of route i to entry point of route j)
        distance_matrix = self._build_distance_matrix(exit_points, entry_points)
        
        # Solve TSP to find optimal order
        method = self._select_method(collection.route_count)
        optimal_order = self._solve_tsp(distance_matrix, method)
        
        # Log the optimization result
        original_distance = self._calculate_total_distance(
            list(range(collection.route_count)), distance_matrix
        )
        optimized_distance = self._calculate_total_distance(optimal_order, distance_matrix)
        improvement = (original_distance - optimized_distance) / original_distance * 100
        logger.info(
            f"Route sequence optimized: "
            f"original_distance={original_distance:.2f}m, "
            f"optimized_distance={optimized_distance:.2f}m, "
            f"improvement={improvement:.1f}%"
        )
        
        # Create new collection with optimized order
        return collection.reorder(optimal_order)
    
    def _extract_route_endpoints(
        self, collection: RouteCollection
    ) -> Tuple[List[Waypoint], List[Waypoint]]:
        """Extract entry and exit waypoints for each route.
        
        Uses metadata if available, otherwise uses first/last waypoint of each route.
        
        Args:
            collection: RouteCollection to extract endpoints from
            
        Returns:
            Tuple of (entry_points, exit_points) lists
        """
        entry_points = []
        exit_points = []
        
        for i, route in enumerate(collection.routes):
            if not route:
                # Empty route, use origin
                empty_wp = Waypoint(0, 0, 0, 0, -90, 5, "hover")
                entry_points.append(empty_wp)
                exit_points.append(empty_wp)
                continue
            
            # Use metadata if available
            if collection.metadata and i < len(collection.metadata):
                metadata = collection.metadata[i]
                entry = metadata.entry_waypoint if metadata.entry_waypoint else route[0]
                exit = metadata.exit_waypoint if metadata.exit_waypoint else route[-1]
            else:
                # Default: first and last waypoint
                entry = route[0]
                exit = route[-1]
            
            entry_points.append(entry)
            exit_points.append(exit)
        
        return entry_points, exit_points
    
    def _build_distance_matrix(
        self,
        exit_points: List[Waypoint],
        entry_points: List[Waypoint]
    ) -> np.ndarray:
        """Build distance matrix from exit points to entry points.
        
        The distance matrix represents the cost of transitioning from
        route i (at its exit point) to route j (at its entry point).
        
        Args:
            exit_points: List of exit waypoints for each route
            entry_points: List of entry waypoints for each route
            
        Returns:
            NxN distance matrix where matrix[i,j] is distance from
            exit of route i to entry of route j
        """
        n = len(exit_points)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 0.0
                else:
                    matrix[i, j] = self._calculate_distance(exit_points[i], entry_points[j])
        
        return matrix
    
    def _calculate_distance(self, wp1: Waypoint, wp2: Waypoint) -> float:
        """Calculate Euclidean distance between two waypoints.
        
        Args:
            wp1: First waypoint
            wp2: Second waypoint
            
        Returns:
            Euclidean distance in meters
        """
        return math.sqrt(
            (wp1.x - wp2.x) ** 2 +
            (wp1.y - wp2.y) ** 2 +
            (wp1.z - wp2.z) ** 2
        )
    
    def _select_method(self, n_routes: int) -> str:
        """Select TSP solving method based on problem size.
        
        Args:
            n_routes: Number of routes to optimize
            
        Returns:
            Selected method name
        """
        if self.method != "auto":
            return self.method
        
        # For route sequencing, greedy is usually sufficient
        # and much faster than exact methods
        if n_routes < 10:
            return "greedy"
        else:
            return "greedy"
    
    def _solve_tsp(self, distance_matrix: np.ndarray, method: str) -> List[int]:
        """Solve TSP to find optimal route order.
        
        Args:
            distance_matrix: NxN matrix of transition distances
            method: Solving method ("greedy" or "ortools")
            
        Returns:
            List of route indices in optimal order
        """
        if method == "greedy":
            return self._solve_greedy(distance_matrix)
        elif method == "ortools":
            return self._solve_ortools(distance_matrix)
        else:
            return self._solve_greedy(distance_matrix)
    
    def _solve_greedy(self, distance_matrix: np.ndarray) -> List[int]:
        """Solve TSP using greedy nearest neighbor algorithm.
        
        Starts from route 0 and always picks the nearest unvisited route.
        Fast and works well for route sequencing.
        
        Args:
            distance_matrix: NxN distance matrix
            
        Returns:
            List of indices in visitation order
        """
        n = len(distance_matrix)
        if n == 0:
            return []
        if n == 1:
            return [0]
        
        # Start from route 0
        unvisited = set(range(1, n))
        current = 0
        order = [current]
        
        while unvisited:
            # Find nearest unvisited route
            nearest = min(unvisited, key=lambda i: distance_matrix[current][i])
            order.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return order
    
    def _solve_ortools(self, distance_matrix: np.ndarray) -> List[int]:
        """Solve TSP using Google OR-Tools.
        
        Provides near-optimal solutions for small to medium problems.
        
        Args:
            distance_matrix: NxN distance matrix
            
        Returns:
            List of indices in visitation order
        """
        try:
            from ortools.constraint_solver import routing_enums_pb2
            from ortools.constraint_solver import pywrapcp
        except ImportError:
            logger.warning("OR-Tools not available, falling back to greedy")
            return self._solve_greedy(distance_matrix)
        
        # Scale to integers (OR-Tools requires integers)
        scale = 1000
        dist_int = (distance_matrix * scale).astype(int)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(len(dist_int), 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        # Define distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dist_int[from_node][to_node]
        
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
        search_parameters.time_limit.FromSeconds(2)
        
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
            logger.warning("OR-Tools failed to find solution, falling back to greedy")
            return self._solve_greedy(distance_matrix)
    
    def _calculate_total_distance(self, order: List[int], distance_matrix: np.ndarray) -> float:
        """Calculate total transition distance for a given order.
        
        Args:
            order: List of route indices in order
            distance_matrix: NxN distance matrix
            
        Returns:
            Total transition distance
        """
        total = 0.0
        for i in range(len(order) - 1):
            total += distance_matrix[order[i]][order[i + 1]]
        return total
