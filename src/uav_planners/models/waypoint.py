"""Waypoint model for UAV path planning."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class WaypointAction(Enum):
    """Actions that can be performed at a waypoint."""
    HOVER = "hover"
    SHOOT = "shoot"
    LAND = "land"
    TAKEOFF = "takeoff"


@dataclass
class Waypoint:
    """A single point in the UAV trajectory.

    Uses ENU coordinate system (East-North-Up):
    - x: East direction in meters
    - y: North direction in meters
    - z: Altitude above takeoff point in meters

    Attributes:
        x: East coordinate in meters
        y: North coordinate in meters
        z: Altitude in meters (above takeoff)
        heading_deg: Yaw angle in degrees (0=North, 90=East)
        gimbal_pitch_deg: Camera pitch in degrees (-90=down, 0=horizontal)
        speed_ms: Target speed in meters per second
        action: Action to perform at this waypoint
        dwell_time_s: Time to hover in seconds (for HOVER action)
        is_keypoint: Whether this is a key viewpoint (not interpolated)
        waypoint_type: Type of waypoint for visualization ("coverage" | "transition" | "truncate")
        parent_route_id: ID of the parent route for transition waypoints
    """
    x: float
    y: float
    z: float
    heading_deg: float = 0.0
    gimbal_pitch_deg: float = 0.0
    speed_ms: float = 5.0
    action: WaypointAction = field(default_factory=lambda: WaypointAction.HOVER)
    dwell_time_s: float = 0.0
    is_keypoint: bool = False
    waypoint_type: str = "coverage"
    parent_route_id: Optional[int] = None

    def __post_init__(self):
        """Validate waypoint_type if provided."""
        valid_types = ("coverage", "transition", "truncate")
        if self.waypoint_type not in valid_types:
            raise ValueError(
                f"Invalid waypoint_type: {self.waypoint_type}. "
                f"Must be one of: {valid_types}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert waypoint to dictionary representation.

        Returns:
            Dictionary containing all waypoint fields
        """
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'heading_deg': self.heading_deg,
            'gimbal_pitch_deg': self.gimbal_pitch_deg,
            'speed_ms': self.speed_ms,
            'action': self.action.value if isinstance(self.action, WaypointAction) else self.action,
            'dwell_time_s': self.dwell_time_s,
            'is_keypoint': self.is_keypoint,
            'waypoint_type': self.waypoint_type,
            'parent_route_id': self.parent_route_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Waypoint':
        """Create a Waypoint from a dictionary.

        Args:
            data: Dictionary containing waypoint fields

        Returns:
            New Waypoint instance
        """
        # Handle action conversion from string
        action = data.get('action', 'hover')
        if isinstance(action, str):
            action = WaypointAction(action)

        return cls(
            x=float(data['x']),
            y=float(data['y']),
            z=float(data['z']),
            heading_deg=float(data.get('heading_deg', 0.0)),
            gimbal_pitch_deg=float(data.get('gimbal_pitch_deg', 0.0)),
            speed_ms=float(data.get('speed_ms', 5.0)),
            action=action,
            dwell_time_s=float(data.get('dwell_time_s', 0.0)),
            is_keypoint=bool(data.get('is_keypoint', False)),
            waypoint_type=str(data.get('waypoint_type', 'coverage')),
            parent_route_id=data.get('parent_route_id'),
        )
