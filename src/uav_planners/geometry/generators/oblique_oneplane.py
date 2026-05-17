"""Oblique one-plane boustrophedon trajectory generator."""

import math
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon

from ..base_generator import BaseGeometryGenerator, GeneratorConfig
from ..registry import register_generator
from ..generators.boustrophedon import BoustrophedonGenerator
from ...models.waypoint import Waypoint, WaypointAction
from ...models.camera import Camera
from ...models.pointcloud import PointCloud


@register_generator("oblique_oneplane")
class ObliqueOnePlaneGenerator(BaseGeometryGenerator):
    """Boustrophedon scan over a single oblique plane."""

    def __init__(self) -> None:
        self._boustrophedon = BoustrophedonGenerator()

    @property
    def name(self) -> str:
        return "oblique_oneplane"

    def generate(
        self,
        pointcloud: PointCloud,
        camera: Camera,
        config: GeneratorConfig,
    ) -> List[Waypoint]:
        _ = pointcloud
        polygon_xyz = getattr(config, "oneplane_polygon_xyz", None)
        if polygon_xyz is None:
            raise ValueError("oneplane_polygon_xyz is required for oblique_oneplane")

        points = np.asarray(polygon_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("oneplane_polygon_xyz must be a list of 3D points")
        if points.shape[0] < 3:
            raise ValueError("oneplane_polygon_xyz must contain at least 3 points")

        origin, normal = self._fit_plane(points)
        normal_sign = float(getattr(config, "oneplane_face_normal_sign", -1.0))
        if normal_sign == 0.0:
            raise ValueError("oneplane_face_normal_sign must be non-zero")
        facing_dir = self._normalize(normal * normal_sign)

        axis_u, axis_v = self._build_plane_axes(points, origin, normal)
        polygon_2d = self._project_polygon(points, origin, axis_u, axis_v)

        if polygon_2d.is_empty or polygon_2d.area < 1e-6:
            raise ValueError("oneplane_polygon_xyz produces an empty polygon")

        distance = self._resolve_flight_distance(config)
        track_spacing = self._boustrophedon._compute_track_spacing(
            distance, camera, float(config.side_overlap)
        )
        scan_angle = self._boustrophedon._select_scan_direction(polygon_2d)
        scan_lines = self._boustrophedon._generate_scan_lines(polygon_2d, scan_angle, track_spacing)

        heading_offset = float(getattr(config, "oneplane_heading_yaw_offset_deg", 0.0))
        heading_base, pitch_deg = self._direction_to_heading_pitch(facing_dir)

        waypoints: List[Waypoint] = []
        for index, line in enumerate(scan_lines):
            coords = list(line.coords)
            if index % 2 == 1:
                coords = coords[::-1]
            for u_coord, v_coord in coords:
                point_on_plane = origin + axis_u * float(u_coord) + axis_v * float(v_coord)
                position = point_on_plane + facing_dir * distance
                heading = (heading_base + heading_offset) % 360.0

                waypoints.append(
                    Waypoint(
                        x=float(position[0]),
                        y=float(position[1]),
                        z=float(position[2]),
                        heading_deg=heading,
                        gimbal_pitch_deg=pitch_deg,
                        speed_ms=config.speed_ms,
                        action=WaypointAction.SHOOT,
                        dwell_time_s=0.0,
                        is_keypoint=True,
                    )
                )

        return waypoints

    @staticmethod
    def _fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit plane from points using SVD."""
        origin = np.mean(points, axis=0)
        centered = points - origin
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            raise ValueError("Cannot fit plane from collinear points")
        return origin, normal / norm

    @staticmethod
    def _build_plane_axes(
        points: np.ndarray,
        origin: np.ndarray,
        normal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build orthonormal axes on plane."""
        edge = points[1] - points[0]
        edge = edge - normal * float(np.dot(edge, normal))
        if np.linalg.norm(edge) < 1e-6:
            edge = np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        if np.linalg.norm(edge) < 1e-6:
            edge = np.cross(normal, np.array([1.0, 0.0, 0.0], dtype=np.float64))

        axis_u = edge / float(np.linalg.norm(edge))
        axis_v = np.cross(normal, axis_u)
        axis_v = axis_v / float(np.linalg.norm(axis_v))
        return axis_u, axis_v

    @staticmethod
    def _project_polygon(
        points: np.ndarray,
        origin: np.ndarray,
        axis_u: np.ndarray,
        axis_v: np.ndarray,
    ) -> Polygon:
        coords_2d = []
        for point in points:
            delta = point - origin
            coords_2d.append((float(np.dot(delta, axis_u)), float(np.dot(delta, axis_v))))

        polygon = Polygon(coords_2d)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        return polygon

    @staticmethod
    def _resolve_flight_distance(config: GeneratorConfig) -> float:
        global_distance = getattr(config, "global_distance_m", None)
        if global_distance is not None and float(global_distance) > 0:
            return float(global_distance)
        oblique_dst = float(getattr(config, "oblique_dst_srf", 5.0))
        return max(1.0, oblique_dst)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return vector
        return vector / norm

    @staticmethod
    def _direction_to_heading_pitch(direction: np.ndarray) -> Tuple[float, float]:
        """Convert direction vector to heading and gimbal pitch."""
        dir_norm = direction / float(np.linalg.norm(direction))
        heading = float(math.degrees(math.atan2(dir_norm[0], dir_norm[1])) % 360.0)
        pitch = float(-math.degrees(math.asin(max(-1.0, min(1.0, dir_norm[2])))))
        return heading, pitch
