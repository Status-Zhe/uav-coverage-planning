"""Layered ring trajectory generator for shape-wrapped inspection."""

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

from ..base_generator import BaseGeometryGenerator, GeneratorConfig
from ..coordinate_utils import CoordinateTransformer
from ..registry import register_generator
from ...models.camera import Camera
from ...models.pointcloud import PointCloud
from ...models.waypoint import Waypoint, WaypointAction


@register_generator("viewpoint_optimized")
class ViewpointGeneratorOptimized(BaseGeometryGenerator):
    """Generate layered ring sub-routes around pointcloud slices.

    Workflow:
    1. Build tight XY footprint from each height slice.
    2. Expand footprint outward for safety/camera stand-off.
    3. Uniformly sample ring points by fixed arc length.
    4. Output one circular sub-route per layer for Stage4 transition linking.
    """

    def __init__(self, max_points_for_visibility: int = 8000, min_resolution: float = 2.0):
        self.max_points_for_visibility = max_points_for_visibility
        self.min_resolution = min_resolution

    @property
    def name(self) -> str:
        return "viewpoint_optimized"

    def generate(
        self,
        pointcloud: PointCloud,
        camera: Camera,
        config: GeneratorConfig,
    ) -> List[List[Waypoint]]:
        import time

        start_time = time.time()
        transformer = CoordinateTransformer(pointcloud.points)

        if transformer.is_camera_coords():
            local_points = transformer.to_local(pointcloud.points)
            pointcloud = PointCloud(points=local_points)
            # print(f"  Transformed to local coords: Z=[{local_points[:,2].min():.1f}, {local_points[:,2].max():.1f}]")

        full_points = pointcloud.points

        use_full_points = bool(
            getattr(
                config,
                "viewpoint_shape_use_full_points",
                getattr(config, "viewpoint_hull_use_full_points", False),
            )
        )
        pc_sampled = None
        if not use_full_points:
            pc_sampled = self._downsample_pointcloud(pointcloud)
            print(f"  Downsampled from {pointcloud.point_count:,} to {pc_sampled.point_count:,} points")
        hull_points = full_points if use_full_points else pc_sampled.points

        bbox = self._compute_bounding_box(pointcloud)
        layer_routes = self._build_layer_routes(hull_points, bbox, camera, config)

        if not layer_routes:
            fallback_route = self._build_fallback_route(hull_points, bbox, camera, config)
            if fallback_route:
                layer_routes = [fallback_route]

        if transformer.is_camera_coords():
            # print(f"  Converting back to original coordinates (offset: {transformer.z_offset:.2f})")
            for route in layer_routes:
                for waypoint in route:
                    waypoint.z = waypoint.z - transformer.z_offset

        elapsed = time.time() - start_time
        # print(f"  Generated {len(layer_routes)} layered routes")
        # print(f"  Total time: {elapsed:.2f}s")

        return layer_routes

    def _scalar_float(self, value, default: float) -> float:
        """Normalize config values to float, accepting accidental tuple/list wrappers."""
        if value is None:
            return float(default)
        if isinstance(value, (list, tuple)):
            if not value:
                return float(default)
            value = value[0]
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _scalar_int(self, value, default: int) -> int:
        """Normalize config values to int, accepting accidental tuple/list wrappers."""
        if value is None:
            return int(default)
        if isinstance(value, (list, tuple)):
            if not value:
                return int(default)
            value = value[0]
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _build_layer_routes(
        self,
        points: np.ndarray,
        bbox: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        camera: Camera,
        config: GeneratorConfig,
    ) -> List[List[Waypoint]]:
        (min_x, min_y, min_z), (max_x, max_y, max_z) = bbox
        _ = (min_x, min_y, max_x, max_y)

        min_alt_offset = getattr(config, "min_flight_altitude_m", None)
        if min_alt_offset is None:
            min_alt_offset = getattr(config, "viewpoint_min_altitude", 0.0)
        min_fly_z = min_z + max(0.0, self._scalar_float(min_alt_offset, 0.0))
        max_fly_z = max_z + max(0.0, self._scalar_float(getattr(config, "viewpoint_beyond_altitude", 0.0), 0.0))

        layer_step_user = max(self.min_resolution, self._scalar_float(getattr(config, "viewpoint_layer_height_step_m", 2.5), 2.5))
        raw_expand = getattr(config, "global_distance_m", None)
        if raw_expand is None:
            raw_expand = getattr(config, "viewpoint_boundary_expand_m", 4.0)
        expand_distance = max(self._scalar_float(config.safety_distance, 3.0), self._scalar_float(raw_expand, 4.0))
        arc_step_user = max(0.5, self._scalar_float(getattr(config, "viewpoint_ring_arc_step_m", 2.5), 2.5))
        min_points_per_layer = self._scalar_int(getattr(config, "viewpoint_min_points_per_layer", 8), 8)
        layer_order = str(getattr(config, "viewpoint_layer_order", "bottom_up"))
        side_overlap = self._scalar_float(getattr(config, "side_overlap", 0.5), 0.5)
        hull_roundness = max(
            0.0,
            self._scalar_float(
                getattr(
                    config,
                    "viewpoint_shape_roundness_m",
                    getattr(config, "viewpoint_hull_roundness_m", 0.0),
                ),
                0.0,
            ),
        )
        viewpoint_alpha = max(0.1, self._scalar_float(getattr(config, "viewpoint_alpha", 6.0), 6.0))
        shape_method = str(getattr(config, "viewpoint_shape_method", "alpha")).lower()
        area_jump_ratio = max(1.01, self._scalar_float(getattr(config, "viewpoint_layer_area_jump_ratio", 1.6), 1.6))
        insert_max_global = max(0, self._scalar_int(getattr(config, "viewpoint_layer_insert_max_global", 8), 8))

        # print all params
        # print("Viewpoint Optimized Generator Parameters:")
        # print(f"  Altitude: {config.altitude}")
        # print(f"  Speed (m/s): {config.speed_ms}")
        # print(f"  Side Overlap: {config.side_overlap}")
        # print(f"  Front Overlap: {config.front_overlap}")
        # print(f"  Safety Distance: {config.safety_distance}")
        # print(f"  Coverage Threshold: {config.coverage_threshold}")
        # print(f"  Minimum Altitude: {config.viewpoint_min_altitude}")
        # print(f"  Beyond Altitude: {config.viewpoint_beyond_altitude}")
        # print(f"  Layer Height Step (m): {config.viewpoint_layer_height_step_m}")
        # print(f"  Boundary Expand (m): {config.viewpoint_boundary_expand_m}")
        # print(f"  Ring Arc Step (m): {config.viewpoint_ring_arc_step_m}")
        # print(f"  Minimum Points Per Layer: {config.viewpoint_min_points_per_layer}")
        # print(f"  Layer Order: {config.viewpoint_layer_order}")
        # print(f"  Hull Roundness (m): {config.viewpoint_hull_roundness_m}")
        # print(f"  Area Jump Ratio: {config.viewpoint_layer_area_jump_ratio}")
        # print(f"  Insert Max Global: {config.viewpoint_layer_insert_max_global}")

        layer_step, arc_step = self._resolve_sampling_steps(
            camera,
            expand_distance,
            side_overlap,
            layer_step_user,
            arc_step_user,
        )
        if layer_step_user > 0.0:
            layer_step = layer_step_user
        # print(f"  Resolved Layer Step: {layer_step:.2f}m, Arc Step: {arc_step:.2f}m")

        # Use height-slice footprint extraction for concave recesses
        use_height_slice = getattr(config, "viewpoint_use_height_slice", True)
        
        layer_records: List[Tuple[float, Polygon, bool]] = []
        
        if use_height_slice:
            # Height-sliced approach: extract footprints at multiple levels and merge
            sliced_footprints = self._extract_xy_footprint_height_sliced(
                points=points,
                min_z=min_fly_z,
                max_z=max_fly_z,
                layer_step=layer_step,
                config=config,
                alpha=viewpoint_alpha,
                method=shape_method,
            )
            
            # Convert to layer_records format
            for z_level, footprint in sliced_footprints:
                layer_records.append((float(z_level), self._round_footprint(footprint, hull_roundness), False))
            
            # If height-slicing returned too few layers (e.g., no points in some regions),
            # generate additional layers based on layer_step to ensure proper coverage
            if len(layer_records) < 3:
                z_levels = self._compute_layer_z_levels(min_fly_z, max_fly_z, layer_step, "bottom_up")
                existing_z = {fp[0] for fp in layer_records}
                
                global_footprint = None
                last_valid_footprint = None
                
                if layer_records:
                    # Use the last footprint as global (for beyond altitude layers)
                    global_footprint = layer_records[-1][1]
                    last_valid_footprint = global_footprint
                
                for z_level in z_levels:
                    if z_level not in existing_z:
                        # Extract footprint from points at this height level
                        slice_half_window = max(0.8, layer_step * 0.55)
                        slice_points = self._extract_slice_points(points, z_level, slice_half_window)
                        
                        if len(slice_points) >= min_points_per_layer:
                            footprint = self._round_footprint(
                                self._extract_xy_footprint(slice_points[:, :2], alpha=viewpoint_alpha, method=shape_method),
                                hull_roundness,
                            )
                            layer_records.append((float(z_level), footprint, False))
                            last_valid_footprint = footprint
                        elif z_level > max_z and last_valid_footprint is not None:
                            # Use last valid footprint for beyond altitude layers
                            layer_records.append((float(z_level), last_valid_footprint, False))
                        elif z_level > max_z and global_footprint is not None:
                            # Fall back to global footprint
                            layer_records.append((float(z_level), global_footprint, False))
            
            # Sort by z_level to ensure proper order based on layer_order
            if getattr(config, "viewpoint_layer_order", "bottom_up") == "top_down":
                layer_records.sort(key=lambda x: x[0], reverse=True)
            else:
                layer_records.sort(key=lambda x: x[0])
        else:
            # Original approach: extract footprint at each z_level with slice window
            z_levels = self._compute_layer_z_levels(min_fly_z, max_fly_z, layer_step, layer_order)
            slice_half_window = max(0.8, layer_step * 0.55)

            layer_records: List[Tuple[float, Polygon, bool]] = []
            last_valid_footprint: Optional[Polygon] = None
            global_footprint = self._round_footprint(
                self._extract_xy_footprint(points[:, :2], alpha=viewpoint_alpha, method=shape_method),
                hull_roundness,
            )
            for z_level in z_levels:
                slice_points = self._extract_slice_points(points, z_level, slice_half_window)
                if len(slice_points) >= min_points_per_layer:
                    footprint = self._round_footprint(
                        self._extract_xy_footprint(slice_points[:, :2], alpha=viewpoint_alpha, method=shape_method),
                        hull_roundness,
                    )
                    last_valid_footprint = footprint
                elif z_level > max_z and last_valid_footprint is not None:
                    footprint = last_valid_footprint
                elif z_level > max_z:
                    footprint = global_footprint
                else:
                    continue

                layer_records.append((float(z_level), footprint, False))

        if not layer_records:
            return []

        # For height-slice mode, skip interpolated layers since all layers are already extracted
        if use_height_slice:
            augmented_layers = layer_records
        else:
            augmented_layers = self._insert_interpolated_layers(
                layer_records=layer_records,
                area_jump_ratio=area_jump_ratio,
                max_insert_global=insert_max_global,
                camera=camera,
                side_overlap=side_overlap,
                boundary_expand_m=expand_distance,
                vertical_layer_step_m=layer_step,
            )

        routes: List[List[Waypoint]] = []
        for z_level, footprint, is_interpolated in augmented_layers:
            wrapped = footprint.buffer(expand_distance, join_style=2)
            ring_points = self._sample_polygon_exterior(wrapped, arc_step, min_points=min_points_per_layer)
            if len(ring_points) < min_points_per_layer:
                continue

            center = np.array([footprint.centroid.x, footprint.centroid.y], dtype=float)
            gimbal_pitch = -45.0 if is_interpolated else self._compute_gimbal_pitch(z_level, min_z, max_z, max_fly_z)

            route: List[Waypoint] = []
            for x, y in ring_points:
                heading = self._compute_heading_inward_to_polygon(
                    np.array([x, y], dtype=float),
                    footprint,
                    center,
                )
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

    def _insert_interpolated_layers(
        self,
        layer_records: List[Tuple[float, Polygon, bool]],
        area_jump_ratio: float,
        max_insert_global: int,
        camera: Camera,
        side_overlap: float,
        boundary_expand_m: float,
        vertical_layer_step_m: float,
    ) -> List[Tuple[float, Polygon, bool]]:
        if len(layer_records) < 2 or max_insert_global <= 0:
            return layer_records

        augmented: List[Tuple[float, Polygon, bool]] = [layer_records[0]]
        inserted_total = 0

        for idx in range(len(layer_records) - 1):
            z0, poly0, _ = layer_records[idx]
            z1, poly1, _ = layer_records[idx + 1]
            z_gap = abs(z1 - z0)

            area0 = max(float(poly0.area), 1e-6)
            area1 = max(float(poly1.area), 1e-6)
            ratio = max(area0, area1) / min(area0, area1)

            inserts = 0
            if ratio > area_jump_ratio and inserted_total < max_insert_global and min(area0, area1) > 1:
                # print(f"  Area jump detected between layers at Z={z0:.1f}m and Z={z1:.1f}m: ratio={ratio:.2f}, gap={z_gap:.2f}m")
                horizontal_step = self._resolve_horizontal_insert_step(
                    camera,
                    stand_off_distance=max(0.5, z_gap + boundary_expand_m),
                    side_overlap=side_overlap,
                )
                horizontal_gap = self._estimate_horizontal_gap(poly0, poly1)

                max_by_horizontal = max(0, int(math.floor(horizontal_gap / horizontal_step)))
                max_global_remaining = max_insert_global - inserted_total
                inserts = min(max_by_horizontal, max_global_remaining)

                # print(
                #     f"    Horizontal gap: {horizontal_gap:.2f}m, insert_step(horizontal): {horizontal_step:.2f}m, "
                #     f"max_by_horizontal: {max_by_horizontal}"
                # )
                # print(
                #     f"    Vertical gap: {z_gap:.2f}m, high-layer step(vertical): {vertical_layer_step_m:.2f}m, "
                #     f"max_global_remaining: {max_global_remaining}"
                # )
                # print(f"    Planning to insert {inserts} layers")

            z_insert_values: List[float] = []
            if inserts > 0:
                higher_z = max(z0, z1)
                lower_z = min(z0, z1)
                vertical_step = max(1e-6, float(vertical_layer_step_m))
                for insert_idx in range(1, inserts + 1):
                    z_from_high = higher_z - vertical_step * insert_idx
                    if lower_z < z_from_high < higher_z:
                        z_insert = float(z_from_high)
                    else:
                        z_insert = float(higher_z)
                    z_insert_values.append(z_insert)
                if z0 < z1:
                    z_insert_values.sort()
                else:
                    z_insert_values.sort(reverse=True)

            for idx_insert, z_mid in enumerate(z_insert_values, start=1):
                t_insert = idx_insert / float(inserts + 1)
                poly_mid = self._interpolate_footprint(poly0, poly1, t_insert)
                augmented.append((float(z_mid), poly_mid, True))
                inserted_total += 1

            augmented.append((z1, poly1, False))

        return augmented

    def _resolve_horizontal_insert_step(self, camera: Camera, stand_off_distance: float, side_overlap: float) -> float:
        overlap = float(np.clip(side_overlap, 0.0, 0.95))
        fov_h = math.radians(camera.fov_horizontal_deg)
        coverage_w = max(0.5, 2.0 * stand_off_distance * math.tan(fov_h / 2.0))
        return max(0.5, coverage_w * (1.0 - overlap))

    def _estimate_horizontal_gap(self, poly0: Polygon, poly1: Polygon) -> float:
        try:
            gap = float(poly0.exterior.hausdorff_distance(poly1.exterior))
            return max(0.0, gap)
        except Exception:
            c0 = np.array([poly0.centroid.x, poly0.centroid.y], dtype=float)
            c1 = np.array([poly1.centroid.x, poly1.centroid.y], dtype=float)
            return float(np.linalg.norm(c0 - c1))

    def _interpolate_footprint(self, poly0: Polygon, poly1: Polygon, t: float) -> Polygon:
        t_clamped = float(np.clip(t, 0.0, 1.0))

        area0 = max(float(poly0.area), 1e-6)
        area1 = max(float(poly1.area), 1e-6)
        target_area = math.exp((1.0 - t_clamped) * math.log(area0) + t_clamped * math.log(area1))

        base = poly0 if t_clamped <= 0.5 else poly1
        base_area = max(float(base.area), 1e-6)
        scale_factor = math.sqrt(target_area / base_area)

        try:
            scaled = affinity.scale(base, xfact=scale_factor, yfact=scale_factor, origin="centroid")
            c0 = np.array([poly0.centroid.x, poly0.centroid.y], dtype=float)
            c1 = np.array([poly1.centroid.x, poly1.centroid.y], dtype=float)
            target_center = c0 * (1.0 - t_clamped) + c1 * t_clamped
            center_scaled = np.array([scaled.centroid.x, scaled.centroid.y], dtype=float)
            shifted = affinity.translate(
                scaled,
                xoff=float(target_center[0] - center_scaled[0]),
                yoff=float(target_center[1] - center_scaled[1]),
            )

            if shifted.is_empty:
                return poly0 if t_clamped <= 0.5 else poly1

            if shifted.geom_type != "Polygon":
                shifted = shifted.convex_hull
            if not shifted.is_valid:
                shifted = shifted.buffer(0)
            if shifted.is_empty:
                return poly0 if t_clamped <= 0.5 else poly1
            if shifted.geom_type != "Polygon":
                shifted = shifted.convex_hull

            return shifted
        except Exception:
            return poly0 if t_clamped <= 0.5 else poly1

    def _build_fallback_route(
        self,
        points: np.ndarray,
        bbox: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        camera: Camera,
        config: GeneratorConfig,
    ) -> List[Waypoint]:
        (_, _, min_z), (_, _, max_z) = bbox
        min_alt_offset = getattr(config, "min_flight_altitude_m", None)
        if min_alt_offset is None:
            min_alt_offset = getattr(config, "viewpoint_min_altitude", 0.0)
        min_fly_z = min_z + max(0.0, self._scalar_float(min_alt_offset, 0.0))
        max_fly_z = max_z + max(0.0, self._scalar_float(getattr(config, "viewpoint_beyond_altitude", 0.0), 0.0))
        z_mid = (min_fly_z + max_fly_z) * 0.5
        hull_roundness = max(
            0.0,
            self._scalar_float(
                getattr(
                    config,
                    "viewpoint_shape_roundness_m",
                    getattr(config, "viewpoint_hull_roundness_m", 0.0),
                ),
                0.0,
            ),
        )
        expand_distance = max(self._scalar_float(config.safety_distance, 3.0), self._scalar_float(getattr(config, "viewpoint_boundary_expand_m", 4.0), 4.0))
        side_overlap = self._scalar_float(getattr(config, "side_overlap", 0.5), 0.5)
        _, arc_step = self._resolve_sampling_steps(
            camera,
            expand_distance,
            side_overlap,
            max(self.min_resolution, self._scalar_float(getattr(config, "viewpoint_layer_height_step_m", 2.5), 2.5)),
            max(0.5, self._scalar_float(getattr(config, "viewpoint_ring_arc_step_m", 2.5), 2.5)),
        )
        min_points = self._scalar_int(getattr(config, "viewpoint_min_points_per_layer", 8), 8)
        viewpoint_alpha = max(0.1, self._scalar_float(getattr(config, "viewpoint_alpha", 6.0), 6.0))
        shape_method = str(getattr(config, "viewpoint_shape_method", "alpha")).lower()

        footprint = self._round_footprint(
            self._extract_xy_footprint(points[:, :2], alpha=viewpoint_alpha, method=shape_method),
            hull_roundness,
        )
        wrapped = footprint.buffer(expand_distance, join_style=2)
        ring_points = self._sample_polygon_exterior(wrapped, arc_step, min_points=min_points)
        center = np.array([footprint.centroid.x, footprint.centroid.y], dtype=float)
        if not ring_points:
            return []

        route: List[Waypoint] = []
        for x, y in ring_points:
            heading = self._compute_heading_inward_to_polygon(
                np.array([x, y], dtype=float),
                footprint,
                center,
            )
            route.append(
                Waypoint(
                    x=float(x),
                    y=float(y),
                    z=float(z_mid),
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
        """Derive conservative sampling steps from camera FOV and overlap.

        Actual step uses min(user, camera-overlap upper bound).
        """
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
        
        # 点云范围内水平拍摄
        if z_level <= max_z:
            return 0.0
        
        # 超出顶部：线性过渡到-90°
        transition_height = max_fly_z- max_z 
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
        """Limit alpha-shape input size with grid deduplication.

        Keeps runtime predictable for large per-layer slices while preserving
        the geometric outline sufficiently for routing.
        """
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
        """Round polygon corners by buffered smoothing while preserving robustness."""
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
        """Compute inward heading using nearest footprint boundary point.

        This approximates inward normal direction for wrapped ring points.
        Falls back to center direction if boundary projection is degenerate.
        """
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

    def _extract_xy_footprint_height_sliced(
        self,
        points: np.ndarray,
        min_z: float,
        max_z: float,
        layer_step: float,
        config: GeneratorConfig,
        alpha: float = 6.0,
        method: str = "alpha",
    ) -> List[Tuple[float, Polygon]]:
        """
        Extract XY footprint using height-slice approach to handle concave recesses.
        
        The key insight is that small horizontal beams at high levels (like eaves) 
        should not block the view of recessed areas below. By slicing horizontally
        and merging adjacent layers with overlap, we get a more accurate footprint.
        
        Args:
            points: Full point cloud (N, 3)
            min_z, max_z: Z range
            layer_step: Base height between slices
            config: GeneratorConfig with viewpoint_use_height_slice, viewpoint_slice_thickness_pct, viewpoint_overlap_pct
            alpha, method: Parameters for per-layer footprint extraction
        
        Returns:
            List of (z_level, merged_polygon) tuples for each slice level
        """
        # Get config parameters
        slice_thickness_pct = getattr(config, "viewpoint_slice_thickness_pct", 0.2)
        overlap_pct = getattr(config, "viewpoint_overlap_pct", 0.2)
        
        # Slice thickness is a fraction of layer_step (e.g., 20% of 2.5m = 0.5m window)
        slice_window_half = layer_step * slice_thickness_pct / 2  # Half window above/below center
        
        footprint_records: List[Tuple[float, Polygon]] = []
        
        # Generate slices from bottom to top with overlap
        z_level = min_z
        while z_level <= max_z:
            # Define slice window [z_center - half_window, z_center + half_window]
            z_slice_min = z_level - slice_window_half
            z_slice_max = z_level + slice_window_half
            
            # Extract points in this slice
            mask = (points[:, 2] >= z_slice_min) & (points[:, 2] <= z_slice_max)
            slice_points = points[mask]
            
            if len(slice_points) < 3:
                z_level += layer_step * (1 - overlap_pct)  # Move to next slice
                continue
            
            # Project to XY plane and extract footprint
            xy_points = slice_points[:, :2]
            footprint = self._extract_xy_footprint(xy_points, alpha=alpha, method=method)
            
            if footprint is None or footprint.is_empty:
                z_level += layer_step * (1 - overlap_pct)
                continue
            
            footprint_records.append((float(z_level), footprint))
            
            # Move to next slice with overlap
            z_level += layer_step * (1 - overlap_pct)
        
        # If we only have one slice, return it directly
        if len(footprint_records) <= 1:
            return footprint_records
        
        # Merge adjacent footprints based on overlap
        merged_records: List[Tuple[float, Polygon]] = []
        current_z, current_footprint = footprint_records[0]
        
        for next_z, next_footprint in footprint_records[1:]:
            # Calculate overlap ratio
            intersection = current_footprint.intersection(next_footprint)
            
            if intersection.is_empty:
                # No overlap - keep as separate layers
                merged_records.append((current_z, current_footprint))
                current_z, current_footprint = next_z, next_footprint
                continue
            
            # Check if overlap is sufficient (at least overlap_pct of the smaller area)
            min_area = min(current_footprint.area, next_footprint.area)
            overlap_ratio = intersection.area / max(min_area, 1e-6)
            
            if overlap_ratio >= overlap_pct:
                # Merge with previous layer - use average z level
                combined = current_footprint.union(next_footprint)
                current_footprint = combined
            else:
                # Keep as separate layers
                merged_records.append((current_z, current_footprint))
                current_z, current_footprint = next_z, next_footprint

        # Add the last record
        merged_records.append((current_z, current_footprint))
        
        # If merged records are too few (e.g., for stepped footprint with overlapping footprints),
        # generate additional layers based on layer_step to ensure proper coverage
        target_layers = max(3, int(math.ceil((max_z - min_z) / layer_step)) + 1)
        
        if len(merged_records) < target_layers:
            # Generate additional layers at regular intervals
            z_levels = self._compute_layer_z_levels(min_z, max_z, layer_step, "bottom_up")
            existing_z = {fp[0] for fp in merged_records}
            
            global_footprint = None
            if merged_records:
                global_footprint = merged_records[-1][1]
            
            for z_level in z_levels:
                if z_level not in existing_z:
                    # Extract footprint from points at this height level
                    mask = (points[:, 2] >= z_level - slice_window_half) & (points[:, 2] <= z_level + slice_window_half)
                    slice_points = points[mask]
                    
                    if len(slice_points) >= 3:
                        xy_points = slice_points[:, :2]
                        footprint = self._extract_xy_footprint(xy_points, alpha=alpha, method=method)
                        if footprint is not None and not footprint.is_empty:
                            global_footprint = footprint
                    if global_footprint is None:
                        continue
                    
                    merged_records.append((float(z_level), global_footprint))
        
        # Sort by z_level to ensure proper order (bottom-up or top-down based on input)
        merged_records.sort(key=lambda x: x[0])
        
        # If too many layers, keep only the first target_layers
        if len(merged_records) > target_layers:
            merged_records = merged_records[:target_layers]
        
        return merged_records



