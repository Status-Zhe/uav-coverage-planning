"""Adaptive capture point sampling and linear interpolation utilities."""

from __future__ import annotations

import math
from typing import List

import numpy as np
from scipy.spatial import cKDTree

from ..models.camera import Camera
from ..models.pointcloud import PointCloud
from ..models.waypoint import Waypoint


class CapturePointSampler:
    """Generate capture indices for coverage waypoints.

    Spacing is estimated from camera FOV + overlap and adjusted by
    flight speed and local pointcloud distance.
    """

    def compute_base_spacing(
        self,
        camera: Camera,
        altitude_m: float,
        front_overlap: float,
        speed_ms: float,
    ) -> float:
        """Compute baseline capture spacing in meters."""
        overlap = float(np.clip(front_overlap, 0.0, 0.98))
        footprint = 2.0 * altitude_m * math.tan(math.radians(camera.fov_vertical_deg) / 2.0)
        spacing = max(0.8, footprint * (1.0 - overlap))

        speed_factor = float(np.clip(speed_ms / 5.0, 0.8, 1.4))
        return max(0.8, spacing * speed_factor)

    def mark_capture_points(
        self,
        waypoints: List[Waypoint],
        pointcloud: PointCloud,
        camera: Camera,
        altitude_m: float,
        front_overlap: float,
        speed_ms: float,
    ) -> List[int]:
        """Return indices in waypoints that should be capture points."""
        if not waypoints:
            return []
        if len(waypoints) == 1:
            return [0]

        tree = cKDTree(pointcloud.points) if pointcloud.point_count > 0 else None
        base_spacing = self.compute_base_spacing(camera, altitude_m, front_overlap, speed_ms)

        captures = [0]
        last_capture = 0
        traveled = 0.0

        for idx in range(1, len(waypoints)):
            prev = waypoints[idx - 1]
            curr = waypoints[idx]
            segment = math.sqrt(
                (curr.x - prev.x) ** 2
                + (curr.y - prev.y) ** 2
                + (curr.z - prev.z) ** 2
            )
            traveled += segment

            local_spacing = base_spacing
            if tree is not None:
                dist, _ = tree.query([curr.x, curr.y, curr.z], k=1)
                relation = float(np.clip(dist / max(altitude_m, 1.0), 0.6, 1.4))
                local_spacing = base_spacing * relation

            if traveled >= local_spacing:
                captures.append(idx)
                last_capture = idx
                traveled = 0.0

        if last_capture != len(waypoints) - 1:
            captures.append(len(waypoints) - 1)

        return sorted(set(captures))

    def interpolate_waypoints(
        self,
        waypoints: List[Waypoint],
        pointcloud: PointCloud,
        camera: Camera,
        altitude_m: float,
        front_overlap: float,
        speed_ms: float,
        interpolation_factor: float = 0.5,
    ) -> List[Waypoint]:
        """Linearly interpolate waypoints with adaptive spacing.

        The interpolation step is derived from adaptive capture spacing multiplied
        by interpolation_factor.
        """
        if len(waypoints) <= 1:
            return waypoints

        base_spacing = self.compute_base_spacing(camera, altitude_m, front_overlap, speed_ms)
        min_step = max(0.2, base_spacing * float(np.clip(interpolation_factor, 0.05, 3.0)))
        tree = cKDTree(pointcloud.points) if pointcloud.point_count > 0 else None

        dense: List[Waypoint] = [waypoints[0]]

        for index in range(len(waypoints) - 1):
            start = waypoints[index]
            end = waypoints[index + 1]

            midpoint = np.array(
                [(start.x + end.x) / 2.0, (start.y + end.y) / 2.0, (start.z + end.z) / 2.0],
                dtype=float,
            )
            local_step = min_step
            if tree is not None:
                dist, _ = tree.query(midpoint, k=1)
                relation = float(np.clip(dist / max(altitude_m, 1.0), 0.6, 1.4))
                local_step = min_step * relation

            segment_length = math.sqrt(
                (end.x - start.x) ** 2
                + (end.y - start.y) ** 2
                + (end.z - start.z) ** 2
            )
            steps = max(1, int(math.ceil(segment_length / max(0.3, local_step))))

            for step in range(1, steps + 1):
                t = step / steps
                dense.append(
                    Waypoint(
                        x=float(start.x + t * (end.x - start.x)),
                        y=float(start.y + t * (end.y - start.y)),
                        z=float(start.z + t * (end.z - start.z)),
                        heading_deg=float(self._lerp_angle_deg(start.heading_deg, end.heading_deg, t)),
                        gimbal_pitch_deg=float(
                            start.gimbal_pitch_deg + t * (end.gimbal_pitch_deg - start.gimbal_pitch_deg)
                        ),
                        speed_ms=float(start.speed_ms + t * (end.speed_ms - start.speed_ms)),
                        action=end.action,
                        dwell_time_s=0.0,
                        is_keypoint=False,
                        waypoint_type=getattr(start, "waypoint_type", "coverage"),
                        parent_route_id=getattr(start, "parent_route_id", None),
                    )
                )

        return dense

    @staticmethod
    def _lerp_angle_deg(angle_start: float, angle_end: float, t: float) -> float:
        """Interpolate heading along shortest angular path in degrees."""
        start = float(angle_start)
        end = float(angle_end)
        delta = (end - start + 180.0) % 360.0 - 180.0
        value = start + float(t) * delta
        return (value + 180.0) % 360.0 - 180.0
