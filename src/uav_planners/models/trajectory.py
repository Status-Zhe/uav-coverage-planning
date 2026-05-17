"""Trajectory model for UAV path planning."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

from .waypoint import Waypoint


@dataclass
class Trajectory:
    """Complete UAV flight trajectory with waypoints.
    
    Attributes:
        waypoints: Ordered list of waypoints
        planner_name: Name of the algorithm used
        created_at: Timestamp when trajectory was created
    """
    waypoints: List[Waypoint]
    planner_name: str
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def waypoint_count(self) -> int:
        """Number of waypoints in the trajectory."""
        return len(self.waypoints)
    
    @property
    def total_distance_m(self) -> float:
        """Total distance traveled in meters."""
        if len(self.waypoints) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(self.waypoints)):
            p1 = self.waypoints[i - 1]
            p2 = self.waypoints[i]
            dist = np.sqrt(
                (p2.x - p1.x) ** 2 +
                (p2.y - p1.y) ** 2 +
                (p2.z - p1.z) ** 2
            )
            total += dist
        return total
    
    @property
    def estimated_duration_min(self) -> float:
        """Estimated flight duration in minutes (includes dwell times)."""
        if len(self.waypoints) < 2:
            return 0.0
        
        total_seconds = 0.0
        
        for i in range(1, len(self.waypoints)):
            p1 = self.waypoints[i - 1]
            p2 = self.waypoints[i]
            
            # Distance
            dist = np.sqrt(
                (p2.x - p1.x) ** 2 +
                (p2.y - p1.y) ** 2 +
                (p2.z - p1.z) ** 2
            )
            
            # Travel time (use average speed)
            avg_speed = (p1.speed_ms + p2.speed_ms) / 2
            if avg_speed > 0:
                total_seconds += dist / avg_speed
            
            # Dwell time at destination
            total_seconds += p2.dwell_time_s
        
        return total_seconds / 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary for serialization."""
        return {
            "planner_name": self.planner_name,
            "created_at": self.created_at.isoformat(),
            "waypoint_count": self.waypoint_count,
            "total_distance_m": self.total_distance_m,
            "estimated_duration_min": self.estimated_duration_min,
            "waypoints": [
                {
                    "x": wp.x,
                    "y": wp.y,
                    "z": wp.z,
                    "heading_deg": wp.heading_deg,
                    "gimbal_pitch_deg": wp.gimbal_pitch_deg,
                    "speed_ms": wp.speed_ms,
                    "action": wp.action.value,
                    "dwell_time_s": wp.dwell_time_s,
                    "is_keypoint": wp.is_keypoint,
                }
                for wp in self.waypoints
            ]
        }
    
    def to_json(self) -> str:
        """Convert trajectory to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)
