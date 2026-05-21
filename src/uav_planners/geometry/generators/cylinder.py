"""Cylinder trajectory generator for cylindrical structures."""

import math
from typing import List, Tuple
import numpy as np

from ..base_generator import BaseGeometryGenerator, GeneratorConfig
from ..registry import register_generator
from ...models.waypoint import Waypoint, WaypointAction
from ...models.camera import Camera
from ...models.pointcloud import PointCloud


@register_generator("cylinder")
class CylinderGenerator(BaseGeometryGenerator):
    """Cylinder coverage pattern generator.

    Modes:
    - horizontal: stacked rings with vertical connectors
    - vertical: unwrapped boustrophedon sweep around the cylinder
    """

    @property
    def name(self) -> str:
        return "cylinder"

    def generate(
        self,
        pointcloud: PointCloud,
        camera: Camera,
        config: GeneratorConfig,
    ) -> List[Waypoint]:
        cylinder_center_xy = getattr(config, "cylinder_center_xy", None)
        cylinder_radius = getattr(config, "cylinder_radius", None)
        cylinder_start_z = getattr(config, "cylinder_start_z", None)
        cylinder_height = getattr(config, "cylinder_height", None)

        if (
            cylinder_center_xy is not None
            and cylinder_radius is not None
            and cylinder_start_z is not None
            and cylinder_height is not None
        ):
            center_xy = (float(cylinder_center_xy[0]), float(cylinder_center_xy[1]))
            base_radius = float(cylinder_radius)
            z_start = float(cylinder_start_z)
            height = float(cylinder_height)
            z_min = None
            z_max = None
            gimbal_pitch_override = 0.0
        else:
            center_xy, base_radius, height = self._fit_cylinder(pointcloud)
            z_min = float(np.min(pointcloud.points[:, 2]))
            z_max = float(np.max(pointcloud.points[:, 2]))
            gimbal_pitch_override = None
            z_start = None

        flight_radius = self._compute_flight_radius(base_radius, config)
        z_start, z_end = self._resolve_z_bounds(
            z_min,
            z_max,
            config,
            z_start_override=z_start,
            height_override=height,
        )

        vertical_step = self._compute_vertical_step(flight_radius, camera, config)
        horizontal_step = self._compute_horizontal_step(flight_radius, camera, config)

        mode = str(getattr(config, "cylinder_mode", "horizontal")).lower()
        if mode == "vertical":
            return self._generate_vertical_boustrophedon(
                center_xy,
                flight_radius,
                z_start,
                z_end,
                vertical_step,
                horizontal_step,
                config,
                gimbal_pitch_override,
            )

        return self._generate_horizontal_rings(
            center_xy,
            flight_radius,
            z_start,
            z_end,
            vertical_step,
            horizontal_step,
            config,
            gimbal_pitch_override,
        )

    def _fit_cylinder(self, pointcloud: PointCloud) -> Tuple[Tuple[float, float], float, float]:
        points = pointcloud.points

        min_xyz = np.min(points, axis=0)
        max_xyz = np.max(points, axis=0)

        center_x = (min_xyz[0] + max_xyz[0]) / 2
        center_y = (min_xyz[1] + max_xyz[1]) / 2

        width_x = max_xyz[0] - min_xyz[0]
        width_y = max_xyz[1] - min_xyz[1]
        height = max_xyz[2] - min_xyz[2]

        diameter = (width_x + width_y) / 2
        radius = diameter / 2.5

        return (float(center_x), float(center_y)), float(radius), float(height)

    @staticmethod
    def _compute_flight_radius(base_radius: float, config: GeneratorConfig) -> float:
        global_distance = getattr(config, "global_distance_m", None)
        if global_distance is not None:
            return base_radius + float(global_distance)
        print("Warning: global_distance_m not set, using default expansion for flight radius.")
        return base_radius * 1.8

    @staticmethod
    def _compute_vertical_step(
        flight_radius: float,
        camera: Camera,
        config: GeneratorConfig,
    ) -> float:
        fov_v_rad = math.radians(camera.fov_vertical_deg)
        vertical_coverage = 2 * flight_radius * math.tan(fov_v_rad / 2)
        step = vertical_coverage * (1 - config.front_overlap)
        return max(step, 1.0)

    @staticmethod
    def _compute_horizontal_step(
        flight_radius: float,
        camera: Camera,
        config: GeneratorConfig,
    ) -> float:
        fov_h_rad = math.radians(camera.fov_horizontal_deg)
        horizontal_coverage = 2 * flight_radius * math.tan(fov_h_rad / 2)
        step = horizontal_coverage * (1 - config.side_overlap)
        return max(step, 1.0)

    @staticmethod
    def _resolve_z_bounds(
        z_min: float,
        z_max: float,
        config: GeneratorConfig,
        z_start_override: float = None,
        height_override: float = None,
    ) -> Tuple[float, float]:
        if z_start_override is not None and height_override is not None:
            z_start = float(z_start_override)
            z_end = z_start + float(height_override)
            return z_start, z_end

        min_alt_offset = getattr(config, "min_flight_altitude_m", None)
        if min_alt_offset is None:
            min_alt_offset = config.altitude * 0.5
        z_start = float(z_min) + float(min_alt_offset)
        z_end = float(z_max)
        if z_end <= z_start:
            z_start = float(z_min)
            z_end = float(z_max)
        if z_end <= z_start:
            z_end = z_start + 1.0
        return z_start, z_end

    def _generate_horizontal_rings(
        self,
        center_xy: Tuple[float, float],
        flight_radius: float,
        z_start: float,
        z_end: float,
        vertical_step: float,
        horizontal_step: float,
        config: GeneratorConfig,
        gimbal_pitch_override: float,
    ) -> List[Waypoint]:
        ring_count, ring_spacing = self._resolve_ring_params(
            z_start,
            z_end,
            vertical_step,
            config,
        )

        circumference = 2 * math.pi * flight_radius
        points_per_ring = max(12, int(math.ceil(circumference / max(horizontal_step, 0.1))))
        angle_start = math.radians(getattr(config, "cylinder_angle_start_deg", 0.0))

        waypoints: List[Waypoint] = []
        prev_heading = None

        for ring_index in range(ring_count):
            z = z_start + ring_index * ring_spacing
            if ring_index == ring_count - 1:
                z = z_end

            for i in range(points_per_ring + 1):
                theta = angle_start + 2 * math.pi * (i / points_per_ring)
                x, y = self._point_on_circle(center_xy, flight_radius, theta)
                heading = self._heading_to_center(center_xy, x, y, prev_heading)
                prev_heading = heading

                waypoints.append(
                    self._build_waypoint(x, y, z, heading, config, gimbal_pitch_override)
                )

            if ring_index < ring_count - 1:
                z_next = z_start + (ring_index + 1) * ring_spacing
                if z_next != z:
                    x, y = self._point_on_circle(center_xy, flight_radius, angle_start + 2 * math.pi)
                    heading = self._heading_to_center(center_xy, x, y, prev_heading)
                    prev_heading = heading
                    waypoints.append(
                        self._build_waypoint(x, y, z_next, heading, config, gimbal_pitch_override)
                    )

        return waypoints

    def _generate_vertical_boustrophedon(
        self,
        center_xy: Tuple[float, float],
        flight_radius: float,
        z_start: float,
        z_end: float,
        vertical_step: float,
        horizontal_step: float,
        config: GeneratorConfig,
        gimbal_pitch_override: float,
    ) -> List[Waypoint]:
        strip_count, angle_step = self._resolve_strip_params(
            flight_radius,
            horizontal_step,
            config,
        )
        angle_start = math.radians(getattr(config, "cylinder_angle_start_deg", 0.0))

        z_points = self._sample_z_points(z_start, z_end, vertical_step)

        waypoints: List[Waypoint] = []
        prev_heading = None

        for strip_index in range(strip_count):
            angle = angle_start + strip_index * angle_step
            z_sequence = z_points if strip_index % 2 == 0 else list(reversed(z_points))

            for z in z_sequence:
                x, y = self._point_on_circle(center_xy, flight_radius, angle)
                heading = self._heading_to_center(center_xy, x, y, prev_heading)
                prev_heading = heading
                waypoints.append(
                    self._build_waypoint(x, y, z, heading, config, gimbal_pitch_override)
                )

            if strip_index < strip_count - 1:
                next_angle = angle + angle_step
                connector_z = z_sequence[-1]
                prev_heading = self._append_arc_segment(
                    waypoints,
                    center_xy,
                    flight_radius,
                    angle,
                    next_angle,
                    connector_z,
                    angle_step,
                    config,
                    gimbal_pitch_override,
                    prev_heading,
                )

        return waypoints

    @staticmethod
    def _resolve_ring_params(
        z_start: float,
        z_end: float,
        vertical_step: float,
        config: GeneratorConfig,
    ) -> Tuple[int, float]:
        height = max(z_end - z_start, 0.0)
        ring_count = getattr(config, "cylinder_ring_count", None)
        if ring_count is not None:
            ring_count = max(int(ring_count), 1)
            ring_spacing = height / max(ring_count - 1, 1)
            return ring_count, ring_spacing

        ring_spacing = getattr(config, "cylinder_ring_spacing_m", None)
        if ring_spacing is not None:
            ring_spacing = max(float(ring_spacing), 0.1)
        else:
            ring_spacing = max(float(vertical_step), 0.1)

        ring_count = max(int(math.floor(height / ring_spacing)) + 1, 1)
        if ring_count > 1:
            ring_spacing = height / (ring_count - 1)
        return ring_count, ring_spacing

    @staticmethod
    def _resolve_strip_params(
        flight_radius: float,
        horizontal_step: float,
        config: GeneratorConfig,
    ) -> Tuple[int, float]:
        strip_count = getattr(config, "cylinder_strip_count", None)
        if strip_count is not None:
            strip_count = max(int(strip_count), 2)
            return strip_count, (2 * math.pi) / strip_count

        strip_spacing = getattr(config, "cylinder_strip_spacing_m", None)
        if strip_spacing is not None:
            strip_spacing = max(float(strip_spacing), 0.1)
            angle_step = strip_spacing / max(flight_radius, 0.1)
        else:
            angle_step = max(float(horizontal_step) / max(flight_radius, 0.1), 0.1)

        strip_count = max(int(math.ceil((2 * math.pi) / angle_step)), 2)
        angle_step = (2 * math.pi) / strip_count
        return strip_count, angle_step

    @staticmethod
    def _sample_z_points(z_start: float, z_end: float, vertical_step: float) -> List[float]:
        if z_end <= z_start:
            return [z_start]
        step = max(float(vertical_step), 0.1)
        count = max(int(math.floor((z_end - z_start) / step)) + 1, 2)
        z_points = [z_start + i * step for i in range(count - 1)]
        z_points.append(z_end)
        return z_points

    @staticmethod
    def _point_on_circle(
        center_xy: Tuple[float, float],
        radius: float,
        theta: float,
    ) -> Tuple[float, float]:
        cx, cy = center_xy
        return cx + radius * math.cos(theta), cy + radius * math.sin(theta)

    @classmethod
    def _heading_to_center(
        cls,
        center_xy: Tuple[float, float],
        x: float,
        y: float,
        prev_heading: float,
    ) -> float:
        cx, cy = center_xy
        dx = cx - x
        dy = cy - y
        raw_heading = math.degrees(math.atan2(dx, dy))
        return cls._unwrap_heading_to_previous(raw_heading, prev_heading)

    @staticmethod
    def _build_waypoint(
        x: float,
        y: float,
        z: float,
        heading: float,
        config: GeneratorConfig,
        gimbal_pitch_override: float,
    ) -> Waypoint:
        gimbal_pitch = gimbal_pitch_override
        if gimbal_pitch is None:
            gimbal_pitch = -45.0

        return Waypoint(
            x=float(x),
            y=float(y),
            z=float(z),
            heading_deg=float(heading),
            gimbal_pitch_deg=float(gimbal_pitch),
            speed_ms=config.speed_ms,
            action=WaypointAction.SHOOT,
            dwell_time_s=0.0,
            is_keypoint=True,
        )

    def _append_arc_segment(
        self,
        waypoints: List[Waypoint],
        center_xy: Tuple[float, float],
        radius: float,
        angle_from: float,
        angle_to: float,
        z: float,
        angle_step: float,
        config: GeneratorConfig,
        gimbal_pitch_override: float,
        prev_heading: float,
    ) -> float:
        angle_diff = angle_to - angle_from
        steps = max(int(math.ceil(abs(angle_diff) / max(angle_step, 1e-6))) + 1, 2)

        for i in range(1, steps + 1):
            t = i / steps
            theta = angle_from + angle_diff * t
            x, y = self._point_on_circle(center_xy, radius, theta)
            heading = self._heading_to_center(center_xy, x, y, prev_heading)
            prev_heading = heading
            waypoints.append(
                self._build_waypoint(x, y, z, heading, config, gimbal_pitch_override)
            )
        return prev_heading

    @staticmethod
    def _unwrap_heading_to_previous(raw_heading_deg: float, previous_heading_deg: float) -> float:
        if previous_heading_deg is None:
            return float(raw_heading_deg)

        candidate = float(raw_heading_deg)
        while candidate - previous_heading_deg > 180.0:
            candidate -= 360.0
        while candidate - previous_heading_deg < -180.0:
            candidate += 360.0
        return candidate
