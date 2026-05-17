"""Safety corridor calculation for path planning."""

import math
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

from ..models.waypoint import Waypoint


@dataclass
class CorridorSegment:
    """A segment of a safety corridor."""
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    radius: float
    corridor_type: str  # "cylinder" or "box"


class SafetyCorridor:
    """Safety corridor for path validation.
    
    Supports multiple corridor types:
    - "cylinder": Cylindrical corridor around path segment
    - "box": Axis-aligned bounding box corridor
    
    Example:
        corridor = SafetyCorridor(corridor_type="cylinder", width=5.0)
        segments = corridor.compute_corridor(waypoints)
    """
    
    def __init__(self, corridor_type: str = "cylinder", width: float = 2.0):
        """Initialize safety corridor.
        
        Args:
            corridor_type: Type of corridor ("cylinder" or "box")
            width: Corridor width/radius in meters
            
        Raises:
            ValueError: If corridor_type is not supported
        """
        if corridor_type not in ["cylinder", "box"]:
            raise ValueError(
                f"Unknown corridor type: {corridor_type}. "
                f"Supported: cylinder, box"
            )
        
        self.corridor_type = corridor_type
        self.width = width
    
    def compute_corridor(
        self,
        waypoints: List[Waypoint]
    ) -> List[CorridorSegment]:
        """Compute safety corridor for a waypoint sequence.
        
        Args:
            waypoints: Ordered list of waypoints
            
        Returns:
            List of corridor segments
        """
        if len(waypoints) < 2:
            return []
        
        segments = []
        
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            
            segment = CorridorSegment(
                start=(wp1.x, wp1.y, wp1.z),
                end=(wp2.x, wp2.y, wp2.z),
                radius=self.width,
                corridor_type=self.corridor_type
            )
            segments.append(segment)
        
        return segments
    
    def check_point_in_corridor(
        self,
        point: Tuple[float, float, float],
        segment: CorridorSegment
    ) -> bool:
        """Check if a point lies within a corridor segment.
        
        Args:
            point: Point to check (x, y, z)
            segment: Corridor segment
            
        Returns:
            True if point is inside corridor
        """
        if segment.corridor_type == "cylinder":
            return self._point_in_cylinder(point, segment)
        else:
            return self._point_in_box(point, segment)
    
    def _point_in_cylinder(
        self,
        point: Tuple[float, float, float],
        segment: CorridorSegment
    ) -> bool:
        """Check if point is inside cylindrical corridor.
        
        Args:
            point: Point to check
            segment: Cylindrical segment
            
        Returns:
            True if inside cylinder
        """
        px, py, pz = point
        sx, sy, sz = segment.start
        ex, ey, ez = segment.end
        
        # Vector from start to end
        dx = ex - sx
        dy = ey - sy
        dz = ez - sz
        
        # Vector from start to point
        px_s = px - sx
        py_s = py - sy
        pz_s = pz - sz
        
        # Segment length squared
        seg_len_sq = dx*dx + dy*dy + dz*dz
        
        if seg_len_sq == 0:
            # Degenerate segment, check distance to start point
            dist_sq = px_s*px_s + py_s*py_s + pz_s*pz_s
            return dist_sq <= segment.radius * segment.radius
        
        # Project point onto line segment
        t = max(0.0, min(1.0, (px_s*dx + py_s*dy + pz_s*dz) / seg_len_sq))
        
        # Closest point on segment
        closest_x = sx + t * dx
        closest_y = sy + t * dy
        closest_z = sz + t * dz
        
        # Distance to closest point
        dist_sq = (
            (px - closest_x)**2 +
            (py - closest_y)**2 +
            (pz - closest_z)**2
        )
        
        return dist_sq <= segment.radius * segment.radius
    
    def _point_in_box(
        self,
        point: Tuple[float, float, float],
        segment: CorridorSegment
    ) -> bool:
        """Check if point is inside box corridor.
        
        Args:
            point: Point to check
            segment: Box segment
            
        Returns:
            True if inside box
        """
        px, py, pz = point
        sx, sy, sz = segment.start
        ex, ey, ez = segment.end
        
        # Bounding box with padding
        min_x = min(sx, ex) - segment.radius
        max_x = max(sx, ex) + segment.radius
        min_y = min(sy, ey) - segment.radius
        max_y = max(sy, ey) + segment.radius
        min_z = min(sz, ez) - segment.radius
        max_z = max(sz, ez) + segment.radius
        
        return (
            min_x <= px <= max_x and
            min_y <= py <= max_y and
            min_z <= pz <= max_z
        )
    
    def validate_trajectory(
        self,
        waypoints: List[Waypoint],
        obstacle_points: np.ndarray
    ) -> Tuple[bool, List[int]]:
        """Validate that trajectory stays within corridor.
        
        Args:
            waypoints: Planned trajectory
            obstacle_points: Points to check against (N x 3 array)
            
        Returns:
            Tuple of (is_valid, list of violating waypoint indices)
        """
        segments = self.compute_corridor(waypoints)
        violations = []
        
        for i, wp in enumerate(waypoints):
            wp_point = (wp.x, wp.y, wp.z)
            
            # Check if waypoint is in any segment
            in_corridor = any(
                self.check_point_in_corridor(wp_point, seg)
                for seg in segments
            )
            
            if not in_corridor:
                violations.append(i)
        
        return len(violations) == 0, violations
