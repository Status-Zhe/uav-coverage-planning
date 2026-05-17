"""Result container for coverage planning."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..models.waypoint import Waypoint
from ..models.trajectory import Trajectory
from ..models.route_collection import RouteCollection
from ..utils.waypoint_frame_converter import maybe_convert_waypoint_pose


@dataclass
class PlannerResult:
    """Result of a coverage planning operation.
    
    Contains both sparse keypoints and dense interpolated trajectory,
    along with coverage statistics.
    
    Attributes:
        waypoints: Complete trajectory (sparse + interpolated)
        sparse_waypoints: Key viewpoints only (is_keypoint=True)
        dense_trajectory: Full dense trajectory
        routes: Optional RouteCollection preserving multi-route structure
        coverage_report: Coverage statistics
        metadata: Additional planning metadata
        created_at: Timestamp
    """
    waypoints: List[Waypoint]
    sparse_waypoints: List[Waypoint] = field(default_factory=list)
    dense_trajectory: List[Waypoint] = field(default_factory=list)
    routes: Optional[RouteCollection] = None
    coverage_report: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize derived fields if not provided."""
        if not self.sparse_waypoints:
            self.sparse_waypoints = [wp for wp in self.waypoints if wp.is_keypoint]
        
        if not self.dense_trajectory:
            self.dense_trajectory = self.waypoints.copy()
    
    @property
    def keypoint_count(self) -> int:
        """Number of key viewpoints."""
        return len(self.sparse_waypoints)
    
    @property
    def total_waypoints(self) -> int:
        """Total number of waypoints in dense trajectory."""
        return len(self.dense_trajectory)
    
    @property
    def coverage_ratio(self) -> float:
        """Coverage ratio from report (0-1)."""
        return self.coverage_report.get("coverage_ratio", 0.0)
    
    def to_trajectory(self, planner_name: str = "CoveragePlanner") -> Trajectory:
        """Convert to Trajectory model.
        
        Args:
            planner_name: Name of the planner used
            
        Returns:
            Trajectory object
        """
        return Trajectory(
            waypoints=self.dense_trajectory,
            planner_name=planner_name
        )
    
    def export_json(self, filepath: str) -> None:
        """Export result to JSON file.
        
        Args:
            filepath: Output file path
        """
        import json
        
        conversion_state = self._resolve_export_conversion_state()

        # Build waypoints list
        waypoints_list = [
            {
                "x": pose[0],
                "y": pose[1],
                "z": pose[2],
                "heading_deg": pose[3],
                "gimbal_pitch_deg": wp.gimbal_pitch_deg,
                "speed_ms": wp.speed_ms,
                "action": wp.action.value if hasattr(wp.action, 'value') else str(wp.action),
                "dwell_time_s": wp.dwell_time_s,
                "is_keypoint": wp.is_keypoint,
                "waypoint_type": getattr(wp, 'waypoint_type', 'coverage'),
                "parent_route_id": getattr(wp, 'parent_route_id', None),
            }
            for wp in self.dense_trajectory
            for pose in [self._maybe_convert_pose_for_export(wp, conversion_state)]
        ]
        
        # Build routes structure if available
        routes_list = []
        if self.routes:
            for i, route in enumerate(self.routes.routes):
                route_waypoints = [
                    {
                        "x": pose[0], "y": pose[1], "z": pose[2],
                        "heading_deg": pose[3],
                        "is_keypoint": wp.is_keypoint,
                        "waypoint_type": getattr(wp, 'waypoint_type', 'coverage'),
                        "parent_route_id": getattr(wp, 'parent_route_id', None),
                    }
                    for wp in route
                    for pose in [self._maybe_convert_pose_for_export(wp, conversion_state)]
                ]
                # Get route name from metadata if available
                route_name = f"route_{i}"
                if self.routes.metadata and i < len(self.routes.metadata):
                    route_name = self.routes.metadata[i].name
                routes_list.append({
                    "name": route_name,
                    "waypoints": route_waypoints,
                    "count": len(route_waypoints)
                })
        
        # Extract transitions from metadata if available
        transitions_data = self.metadata.get("transitions", [])
        
        # Calculate transition points count
        transition_points = sum(
            t.get("waypoint_count", 0) for t in transitions_data
        )
        
        data = {
            "created_at": self.created_at.isoformat(),
            "metadata": {**self.metadata, "export_frame_applied": conversion_state["applied_frame"]},
            "coverage_report": self.coverage_report,
            "keypoint_count": self.keypoint_count,
            "total_waypoints": self.total_waypoints,
            "transition_points": transition_points,
            "rrt_transition_points": transition_points,
            "waypoints": waypoints_list,
            "routes": routes_list,
            "transitions": transitions_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_csv(self, filepath: str) -> None:
        """Export waypoints to CSV file.
        
        Args:
            filepath: Output file path
        """
        import csv
        conversion_state = self._resolve_export_conversion_state()
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    'x', 'y', 'z', 'heading_deg', 'gimbal_pitch_deg',
                    'speed_ms', 'action', 'dwell_time_s', 'is_keypoint',
                    'waypoint_type', 'parent_route_id'
                ])
                # Write data
                for wp in self.dense_trajectory:
                    x, y, z, heading = self._maybe_convert_pose_for_export(wp, conversion_state)
                    writer.writerow([
                        x, y, z,
                        heading, wp.gimbal_pitch_deg,
                        wp.speed_ms, 
                        wp.action.value if hasattr(wp.action, 'value') else str(wp.action),
                        wp.dwell_time_s, wp.is_keypoint,
                        getattr(wp, 'waypoint_type', 'coverage'),
                        getattr(wp, 'parent_route_id', '')
                    ])
            print(f"Successfully exported {len(self.dense_trajectory)} waypoints")
        except Exception as e:
            print(f"Export failed: {e}")
            raise  # 重新抛出异常

    def _resolve_export_conversion_state(self) -> Dict[str, Any]:
        """Resolve export frame conversion behavior from metadata."""
        export_frame = str(self.metadata.get("waypoint_export_frame", "planning")).lower()
        can_convert = bool(
            maybe_convert_waypoint_pose(
                0.0,
                0.0,
                0.0,
                0.0,
                self.metadata,
            ) is not None
        )

        if export_frame == "planning":
            return {
                "apply_conversion": False,
                "applied_frame": str(self.metadata.get("planning_frame", "planning")),
            }

        if can_convert:
            return {
                "apply_conversion": True,
                "applied_frame": str(self.metadata.get("source_frame", "source")),
            }

        raise ValueError(
            "waypoint export frame 'source' requested, but ENU->ECEF conversion metadata is unavailable"
        )

    def _maybe_convert_pose_for_export(self, wp: Waypoint, conversion_state: Dict[str, Any]) -> tuple:
        """Return export pose (x, y, z, heading_deg) with optional conversion."""
        if not conversion_state.get("apply_conversion", False):
            return wp.x, wp.y, wp.z, wp.heading_deg

        converted = maybe_convert_waypoint_pose(
            wp.x,
            wp.y,
            wp.z,
            wp.heading_deg,
            self.metadata,
        )
        if converted is None:
            return wp.x, wp.y, wp.z, wp.heading_deg
        return converted