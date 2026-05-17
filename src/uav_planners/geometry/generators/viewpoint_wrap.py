"""Standalone wrap-style layered viewpoint generator."""

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

from ..base_generator import BaseGeometryGenerator, GeneratorConfig
from ..coordinate_utils import CoordinateTransformer
from ..registry import register_generator
from ...models.camera import Camera
from ...models.pointcloud import PointCloud
from ...models.waypoint import Waypoint, WaypointAction


@register_generator("viewpoint_wrap")
class ViewpointWrapGenerator(BaseGeometryGenerator):
    """Generate layered wrap routes directly from point-cloud slices.

    This implementation is intentionally standalone and does not inherit from
    `viewpoint_optimized`, so wrap-specific evolution can proceed independently.
    """

    def __init__(self, max_points_for_visibility: int = 12000, min_resolution: float = 2.0):
        self.max_points_for_visibility = max_points_for_visibility
        self.min_resolution = min_resolution

    @property
    def name(self) -> str:
        return "viewpoint_wrap"

    def generate(
        self,
        pointcloud: PointCloud,
        camera: Camera,
        config: GeneratorConfig,
    ) -> List[List[Waypoint]]:
        transformer = CoordinateTransformer(pointcloud.points)

        if transformer.is_camera_coords():
            local_points = transformer.to_local(pointcloud.points)
            pointcloud = PointCloud(points=local_points)

        use_full_points = bool(getattr(config, "viewpoint_shape_use_full_points", False))
        points = pointcloud.points if use_full_points else self._downsample_pointcloud(pointcloud).points

        bbox = self._compute_bounding_box(pointcloud)
        routes = self._build_layer_routes(points, bbox, camera, config)
        if not routes:
            fallback = self._build_fallback_route(points, bbox, camera, config)
            if fallback:
                routes = [fallback]

        if transformer.is_camera_coords():
            for route in routes:
                for waypoint in route:
                    waypoint.z = waypoint.z - transformer.z_offset

        return routes

    def _build_layer_routes(
        self,
        points: np.ndarray,
        bbox: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        camera: Camera,
        config: GeneratorConfig,
    ) -> List[List[Waypoint]]:
        (_, _, min_z), (_, _, max_z) = bbox

        min_alt_offset = getattr(config, "min_flight_altitude_m", None)
        if min_alt_offset is None:
            min_alt_offset = getattr(config, "viewpoint_min_altitude", 0.0)
        min_fly_z = min_z + max(0.0, float(min_alt_offset))
        max_fly_z = max_z + max(0.0, float(getattr(config, "viewpoint_beyond_altitude", 0.0)))

        raw_expand = getattr(config, "global_distance_m", None)
        if raw_expand is None:
            raw_expand = getattr(config, "viewpoint_boundary_expand_m", 4.0)
        expand_distance = max(float(config.safety_distance), float(raw_expand))

        layer_step_user = max(self.min_resolution, float(getattr(config, "viewpoint_layer_height_step_m", 2.5)))
        arc_step_user = max(0.5, float(getattr(config, "viewpoint_ring_arc_step_m", 2.5)))
        side_overlap = float(getattr(config, "side_overlap", 0.5))
        min_points = int(getattr(config, "viewpoint_min_points_per_layer", 8))
        layer_order = str(getattr(config, "viewpoint_layer_order", "bottom_up"))
        alpha = max(0.1, float(getattr(config, "viewpoint_alpha", 6.0)))
        shape_method = str(getattr(config, "viewpoint_shape_method", "alpha")).lower()
        roundness = max(0.0, float(getattr(config, "viewpoint_shape_roundness_m", 0.0)))

        layer_step, arc_step = self._resolve_sampling_steps(
            camera=camera,
            stand_off_distance=expand_distance,
            side_overlap=side_overlap,
            layer_step_user=layer_step_user,
            arc_step_user=arc_step_user,
        )

        z_levels = self._compute_layer_z_levels(min_fly_z, max_fly_z, layer_step, layer_order)
        if not z_levels:
            return []

        global_footprint = self._round_footprint(self._extract_xy_footprint(points[:, :2], alpha, method=shape_method), roundness)
        slice_half_window = max(0.8, layer_step * 0.55)
        last_valid_footprint: Optional[Polygon] = None
        routes: List[List[Waypoint]] = []

        for z_level in z_levels:
            slice_points = self._extract_slice_points(points, z_level, slice_half_window)
            if len(slice_points) >= min_points:
                footprint = self._round_footprint(self._extract_xy_footprint(slice_points[:, :2], alpha, method=shape_method), roundness)
                last_valid_footprint = footprint
            elif z_level > max_z and last_valid_footprint is not None:
                footprint = last_valid_footprint
            elif z_level > max_z:
                footprint = global_footprint
            else:
                continue

            wrapped = footprint.buffer(expand_distance, join_style=2)
            ring_points = self._sample_polygon_exterior(wrapped, arc_step, min_points=min_points)
            if len(ring_points) < min_points:
                continue

            center = np.array([footprint.centroid.x, footprint.centroid.y], dtype=float)
            gimbal_pitch = self._compute_gimbal_pitch(z_level, min_z, max_z, max_fly_z)
            route: List[Waypoint] = []
            for x, y in ring_points:
                source_xy = np.array([x, y], dtype=float)
                heading = self._compute_heading_inward_to_polygon(source_xy, footprint, center)
                route.append(
                    Waypoint(
                        x=float(x),
                        y=float(y),
                        z=float(z_level),
                        heading_deg=float(heading),
                        gimbal_pitch_deg=float(gimbal_pitch),
                        speed_ms=config.speed_ms,
                        action=WaypointAction.SHOOT,
                        dwell_time_s=0.5,
                        is_keypoint=True,
                    )
                )
            routes.append(route)

        return routes

    def _build_fallback_route(
        self,
        points: np.ndarray,
        bbox: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        camera: Camera,
        config: GeneratorConfig,
    ) -> List[Waypoint]:
        (_, _, min_z), (_, _, max_z) = bbox
        z_mid = float((min_z + max_z) * 0.5)

        raw_expand = getattr(config, "global_distance_m", None)
        if raw_expand is None:
            raw_expand = getattr(config, "viewpoint_boundary_expand_m", 4.0)
        expand_distance = max(float(config.safety_distance), float(raw_expand))
        side_overlap = float(getattr(config, "side_overlap", 0.5))
        alpha = max(0.1, float(getattr(config, "viewpoint_alpha", 6.0)))
        shape_method = str(getattr(config, "viewpoint_shape_method", "alpha")).lower()
        roundness = max(0.0, float(getattr(config, "viewpoint_shape_roundness_m", 0.0)))

        _, arc_step = self._resolve_sampling_steps(
            camera=camera,
            stand_off_distance=expand_distance,
            side_overlap=side_overlap,
            layer_step_user=max(self.min_resolution, float(getattr(config, "viewpoint_layer_height_step_m", 2.5))),
            arc_step_user=max(0.5, float(getattr(config, "viewpoint_ring_arc_step_m", 2.5))),
        )
        min_points = int(getattr(config, "viewpoint_min_points_per_layer", 8))

        footprint = self._round_footprint(self._extract_xy_footprint(points[:, :2], alpha, method=shape_method), roundness)
        wrapped = footprint.buffer(expand_distance, join_style=2)
        ring_points = self._sample_polygon_exterior(wrapped, arc_step, min_points=min_points)
        if not ring_points:
            return []

        center = np.array([footprint.centroid.x, footprint.centroid.y], dtype=float)
        route: List[Waypoint] = []
        for x, y in ring_points:
            heading = self._compute_heading_inward_to_polygon(np.array([x, y], dtype=float), footprint, center)
            route.append(
                Waypoint(
                    x=float(x),
                    y=float(y),
                    z=z_mid,
                    heading_deg=float(heading),
                    gimbal_pitch_deg=-35.0,
                    speed_ms=config.speed_ms,
                    action=WaypointAction.SHOOT,
                    dwell_time_s=0.5,
                    is_keypoint=True,
                )
            )

        return route

    def _compute_layer_z_levels(self, min_z: float, max_z: float, step: float, order: str) -> List[float]:
        if max_z - min_z < 1e-6:
            return [float(min_z)]

        z_start = min_z + step * 0.5
        levels = np.arange(z_start, max_z + step * 0.5, step, dtype=float)
        if levels.size == 0:
            levels = np.array([(min_z + max_z) * 0.5], dtype=float)

        if order == "top_down":
            levels = levels[::-1]

        return [float(value) for value in levels]

    def _resolve_sampling_steps(
        self,
        camera: Camera,
        stand_off_distance: float,
        side_overlap: float,
        layer_step_user: float,
        arc_step_user: float,
    ) -> Tuple[float, float]:
        overlap = float(np.clip(side_overlap, 0.0, 0.95))
        fov_h = math.radians(camera.fov_horizontal_deg)
        fov_v = math.radians(camera.fov_vertical_deg)

        coverage_w = max(0.5, 2.0 * stand_off_distance * math.tan(fov_h / 2.0))
        coverage_h = max(0.5, 2.0 * stand_off_distance * math.tan(fov_v / 2.0))

        arc_auto_max = max(0.5, coverage_w * (1.0 - overlap))
        layer_auto_max = max(self.min_resolution, coverage_h * (1.0 - overlap))

        arc_step = max(0.5, min(arc_step_user, arc_auto_max))
        layer_step = max(self.min_resolution, min(layer_step_user, layer_auto_max))
        return layer_step, arc_step

    def _extract_slice_points(self, points: np.ndarray, z_level: float, half_window: float) -> np.ndarray:
        mask = np.abs(points[:, 2] - z_level) <= half_window
        return points[mask]

    def _compute_gimbal_pitch(self, z_level: float, min_z: float, max_z: float, max_fly_z: float) -> float:
        if max_z - min_z < 1e-3:
            return 0.0

        if z_level <= max_z:
            return 0.0

        transition_height = max(1e-6, max_fly_z - max_z)
        above = z_level - max_z
        ratio = min(1.0, above / transition_height)
        return float(-75.0 * ratio)

    def _downsample_pointcloud(self, pointcloud: PointCloud) -> PointCloud:
        if pointcloud.point_count <= self.max_points_for_visibility:
            return pointcloud

        points = pointcloud.points
        bbox_size = (
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min(),
        )
        volume = max(1e-6, bbox_size[0] * bbox_size[1] * bbox_size[2])
        voxel_volume = volume / self.max_points_for_visibility
        voxel_size = voxel_volume ** (1 / 3)

        voxel_indices = np.floor(points / max(voxel_size, 1e-3)).astype(int)
        unique_indices = np.unique(voxel_indices, axis=0)

        downsampled_points = []
        for idx in unique_indices[:self.max_points_for_visibility]:
            voxel_center = (idx + 0.5) * voxel_size
            distances = np.linalg.norm(points - voxel_center, axis=1)
            downsampled_points.append(points[np.argmin(distances)])

        return PointCloud(points=np.array(downsampled_points))

    def _compute_bounding_box(
        self,
        pointcloud: PointCloud,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        points = pointcloud.points
        min_corner = (
            float(np.min(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.min(points[:, 2])),
        )
        max_corner = (
            float(np.max(points[:, 0])),
            float(np.max(points[:, 1])),
            float(np.max(points[:, 2])),
        )
        return min_corner, max_corner

    def _prepare_alpha_points(self, xy_points: np.ndarray, max_points: int = 12000) -> np.ndarray:
        """Limit alpha-shape input size with grid deduplication."""
        points = np.asarray(xy_points, dtype=float)
        if points.shape[0] <= max_points:
            return points

        min_xy = np.min(points, axis=0)
        max_xy = np.max(points, axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)
        grid_bins = max(8.0, float(np.sqrt(max_points)))
        cell_size = float(np.max(span) / grid_bins)
        cell_size = max(cell_size, 1e-4)

        cell_idx = np.floor((points - min_xy) / cell_size).astype(np.int32)
        _, unique_indices = np.unique(cell_idx, axis=0, return_index=True)
        reduced = points[np.sort(unique_indices)]

        if reduced.shape[0] > max_points:
            step = int(np.ceil(reduced.shape[0] / max_points))
            reduced = reduced[::step]

        return reduced

    def _extract_xy_footprint(self, xy_points: np.ndarray, alpha: float = 6.0, method: str = "alpha") -> Polygon:
        alpha_input = self._prepare_alpha_points(xy_points)
        if method == "convex":
            return self._extract_xy_convex_footprint(alpha_input)
        alpha_footprint = self._extract_xy_alpha_shape(alpha_input, alpha)
        if alpha_footprint is not None and not alpha_footprint.is_empty:
            return alpha_footprint
        return self._extract_xy_convex_footprint(alpha_input)

    def _extract_xy_convex_footprint(self, xy_points: np.ndarray) -> Polygon:
        if len(xy_points) < 3:
            min_x, min_y = np.min(xy_points, axis=0)
            max_x, max_y = np.max(xy_points, axis=0)
            return Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])

        try:
            hull = ConvexHull(xy_points)
            return Polygon(xy_points[hull.vertices])
        except Exception:
            min_x, min_y = np.min(xy_points, axis=0)
            max_x, max_y = np.max(xy_points, axis=0)
            return Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])

    def _extract_xy_alpha_shape(self, xy_points: np.ndarray, alpha: float) -> Optional[Polygon]:
        if len(xy_points) < 4:
            return None

        try:
            tri = Delaunay(xy_points)
        except Exception:
            return None

        edge_counter = {}
        alpha_threshold = max(0.1, float(alpha))

        for simplex in tri.simplices:
            pa = xy_points[simplex[0]]
            pb = xy_points[simplex[1]]
            pc = xy_points[simplex[2]]

            a = float(np.linalg.norm(pb - pc))
            b = float(np.linalg.norm(pa - pc))
            c = float(np.linalg.norm(pa - pb))
            semiperimeter = 0.5 * (a + b + c)
            area_sq = semiperimeter * (semiperimeter - a) * (semiperimeter - b) * (semiperimeter - c)
            if area_sq <= 1e-12:
                continue
            area = math.sqrt(area_sq)
            circumradius = (a * b * c) / max(4.0 * area, 1e-12)

            if circumradius > alpha_threshold:
                continue

            edges = (
                tuple(sorted((int(simplex[0]), int(simplex[1])))),
                tuple(sorted((int(simplex[1]), int(simplex[2])))),
                tuple(sorted((int(simplex[2]), int(simplex[0])))),
            )
            for edge in edges:
                edge_counter[edge] = edge_counter.get(edge, 0) + 1

        boundary_edges = [edge for edge, count in edge_counter.items() if count == 1]
        if not boundary_edges:
            return None

        lines = [
            LineString([
                (float(xy_points[i][0]), float(xy_points[i][1])),
                (float(xy_points[j][0]), float(xy_points[j][1])),
            ])
            for i, j in boundary_edges
        ]

        try:
            boundary = unary_union(lines)
            polygons = list(polygonize(boundary))
            if not polygons:
                return None

            merged = unary_union(polygons)
            if merged.is_empty:
                return None

            if merged.geom_type == "Polygon":
                poly = merged
            else:
                geoms = getattr(merged, "geoms", [])
                polygon_geoms = [geom for geom in geoms if geom.geom_type == "Polygon"]
                if not polygon_geoms:
                    return None
                poly = max(polygon_geoms, key=lambda geom: geom.area)

            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                return None
            if poly.geom_type != "Polygon":
                geoms = getattr(poly, "geoms", [])
                polygon_geoms = [geom for geom in geoms if geom.geom_type == "Polygon"]
                if not polygon_geoms:
                    return None
                poly = max(polygon_geoms, key=lambda geom: geom.area)

            return poly
        except Exception:
            return None

    def _round_footprint(self, footprint: Polygon, roundness_m: float) -> Polygon:
        if roundness_m <= 0:
            return footprint

        try:
            rounded = footprint.buffer(roundness_m, join_style=1).buffer(-roundness_m, join_style=1)
            if rounded.is_empty:
                return footprint
            if rounded.geom_type != "Polygon":
                rounded = rounded.convex_hull
            return rounded
        except Exception:
            return footprint

    def _sample_polygon_exterior(self, polygon: Polygon, spacing: float, min_points: int = 8) -> List[Tuple[float, float]]:
        if polygon.is_empty:
            return []

        line = polygon.exterior
        perimeter = max(line.length, spacing)
        sample_count = max(min_points, int(math.ceil(perimeter / max(0.5, spacing))))

        points: List[Tuple[float, float]] = []
        for i in range(sample_count):
            distance = (i / sample_count) * perimeter
            point = line.interpolate(distance)
            points.append((float(point.x), float(point.y)))

        return points

    def _compute_heading_toward(self, source_xy: np.ndarray, target_xy: np.ndarray) -> float:
        dx = float(target_xy[0] - source_xy[0])
        dy = float(target_xy[1] - source_xy[1])
        return float(np.degrees(np.arctan2(dx, dy)))

    def _compute_heading_inward_to_polygon(
        self,
        source_xy: np.ndarray,
        footprint: Polygon,
        fallback_target_xy: np.ndarray,
    ) -> float:
        try:
            if footprint is not None and not footprint.is_empty:
                source_pt = Point(float(source_xy[0]), float(source_xy[1]))
                projected_dist = footprint.exterior.project(source_pt)
                nearest_pt = footprint.exterior.interpolate(projected_dist)
                nearest_xy = np.array([float(nearest_pt.x), float(nearest_pt.y)], dtype=float)
                if np.linalg.norm(nearest_xy - source_xy) > 1e-6:
                    return self._compute_heading_toward(source_xy, nearest_xy)
        except Exception:
            pass

        return self._compute_heading_toward(source_xy, fallback_target_xy)
