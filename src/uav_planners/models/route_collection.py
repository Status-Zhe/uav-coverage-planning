"""RouteCollection model for multi-route UAV path planning."""

from dataclasses import dataclass
from typing import List, Optional

from .waypoint import Waypoint


@dataclass
class RouteMetadata:
    """Metadata for a single route in a RouteCollection.
    
    Attributes:
        label: Human-readable identifier for the route
        color: Optional color for visualization
        entry_waypoint: Optional waypoint where route starts
        exit_waypoint: Optional waypoint where route ends
        original_order: Original index before any reordering
    """
    label: str
    color: Optional[str] = None
    entry_waypoint: Optional[Waypoint] = None
    exit_waypoint: Optional[Waypoint] = None
    original_order: int = 0


@dataclass
class RouteCollection:
    """Collection of multiple routes for UAV coverage planning.
    
    Used to store multiple routes (e.g., nadir, north, east, south, west for 
    oblique photography) with support for independent collision detection,
    TSP optimization of route execution order, and flattened views.
    
    Attributes:
        routes: List of routes, each being a list of waypoints
        labels: Optional list of labels for each route
        metadata: Optional list of metadata for each route
    """
    routes: List[List[Waypoint]]
    labels: Optional[List[str]] = None
    metadata: Optional[List[RouteMetadata]] = None
    
    @property
    def flattened(self) -> List[Waypoint]:
        """Flatten all routes into a single list of waypoints."""
        return [wp for route in self.routes for wp in route]
    
    @property
    def route_count(self) -> int:
        """Number of routes in the collection."""
        return len(self.routes)
    
    def get_route(self, label: str) -> Optional[List[Waypoint]]:
        """Get a route by its label.
        
        Args:
            label: The label of the route to retrieve
            
        Returns:
            The route waypoints if found, None otherwise
        """
        if self.labels is None:
            return None
        try:
            idx = self.labels.index(label)
            return self.routes[idx]
        except ValueError:
            return None
    
    def reorder(self, new_order: List[int]) -> "RouteCollection":
        """Create a new RouteCollection with routes in the specified order.
        
        Args:
            new_order: List of indices specifying the new order
            
        Returns:
            A new RouteCollection with reordered routes
            
        Raises:
            AssertionError: If new_order length doesn't match route_count
        """
        assert len(new_order) == self.route_count
        new_routes = [self.routes[i] for i in new_order]
        new_labels = [self.labels[i] for i in new_order] if self.labels else None
        new_metadata = [self.metadata[i] for i in new_order] if self.metadata else None
        return RouteCollection(new_routes, new_labels, new_metadata)
    
    def rebuild_from_filtered(self, filtered: List[Waypoint]) -> "RouteCollection":
        """Rebuild routes from a filtered flattened list.
        
        This is useful after collision detection removes waypoints from
        the flattened view. Preserves the original route structure.
        
        Args:
            filtered: Filtered list of waypoints (subset of flattened)
            
        Returns:
            A new RouteCollection with routes rebuilt from filtered waypoints
        """
        new_routes = []
        idx = 0
        for route in self.routes:
            count = len(route)
            new_routes.append(filtered[idx:idx + count])
            idx += count
        return RouteCollection(new_routes, self.labels, self.metadata)
    
    def to_continuous_path(self, transitions: List[List[Waypoint]]) -> List[Waypoint]:
        """Convert routes to a continuous path with transition waypoints.
        
        Args:
            transitions: List of waypoint lists connecting each route to the next
            
        Returns:
            Continuous list of waypoints including transitions
        """
        result = []
        for i, route in enumerate(self.routes):
            result.extend(route)
            if i < len(transitions):
                result.extend(transitions[i])
        return result
