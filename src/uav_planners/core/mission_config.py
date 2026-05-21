"""Mission configuration for coverage planning."""

from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path
import numpy as np

from ..models.camera import Camera
from ..utils.camera_scale_conversions import gsd_to_global_distance_m, oblique_dst_srf_to_gsd


@dataclass
class MissionConfig:
    """Configuration for a coverage planning mission.
    
    Attributes:
        pointcloud_path: Input data path (.pcd/.ply point cloud or tileset.json when auto mode)
        camera: Camera specification
        algorithm: Coverage algorithm name ("boustrophedon", "spiral", etc.)
        altitude: Flight altitude above ground (meters)
        speed_ms: Target flight speed (m/s)
        safety_distance: Minimum distance to obstacles (meters)
        coverage_threshold: Required coverage ratio (0-1)
        side_overlap: Side overlap ratio for adjacent strips (0-1)
        front_overlap: Front overlap ratio along flight direction (0-1)
        output_dir: Directory for output files
    """
    pointcloud_path: Optional[str]
    camera: Camera
    algorithm: str = "boustrophedon"
    altitude: float = 50.0
    speed_ms: float = 5.0
    safety_distance: float = 3.0
    coverage_threshold: float = 0.95
    side_overlap: float = 0.7
    front_overlap: float = 0.8
    scan_direction_mode: str = "auto"
    gsd: Optional[float] = None
    global_distance_m: Optional[float] = None
    min_flight_altitude_m: Optional[float] = None

    # 3d-tiles 兼容
    output_dir: str = "./output"
    data_source_type: str = "auto"  # auto | pointcloud_file | tileset
    tileset_path: Optional[str] = None
    tiles_kind: str = "auto"  # pointcloud | model | auto
    tiles_max_points: int = 800000
    tiles_lod_max: Optional[int] = None
    tiles_bbox: Optional[Tuple[float, float, float, float, float, float]] = None
    tiles_output_frame: str = "enu"  # world | enu
    tiles_input_crs: str = "auto"  # auto | ecef
    tiles_enu_origin_ecef: Optional[Tuple[float, float, float]] = None
    tiles_convert_to_ply: bool = False
    tiles_converted_ply_path: Optional[str] = None
    tiles_convert_coord_frame: str = "enu"  # world | enu | centroid | first_point
    tiles_convert_write_ascii: bool = True
    pointcloud_preprocess_enable: bool = True
    region_only_enabled: bool = False
    coverage_area_rect_xy: Optional[Tuple[float, float, float, float]] = None  # (xmin, ymin, xmax, ymax)
    region_ground_z: float = 0.0
    waypoint_export_frame: str = "planning"  # planning | source
    aabb_center: Optional[Tuple[float, float, float]] = None
    aabb_yaw_deg: float = 0.0
    aabb_size: Optional[Tuple[float, float, float]] = None
    aabb_transform: Optional[np.ndarray] = None

    # Spiral circle-only parameters (no pointcloud required)
    spiral_center_xy: Optional[Tuple[float, float]] = None
    spiral_radius: Optional[float] = None
    spiral_start_z: Optional[float] = None
    spiral_height: Optional[float] = None
    spiral_circle_only_enabled: bool = False

    # Cylinder parameters (no pointcloud required)
    cylinder_center_xy: Optional[Tuple[float, float]] = None
    cylinder_radius: Optional[float] = None
    cylinder_start_z: Optional[float] = None
    cylinder_height: Optional[float] = None
    cylinder_mode: str = "horizontal"  # horizontal | vertical
    cylinder_ring_spacing_m: Optional[float] = None
    cylinder_ring_count: Optional[int] = None
    cylinder_strip_spacing_m: Optional[float] = None
    cylinder_strip_count: Optional[int] = None
    cylinder_angle_start_deg: float = 0.0
    cylinder_params_enabled: bool = False

    # Oblique one-plane parameters (no pointcloud required)
    oneplane_polygon_xyz: Optional[List[Tuple[float, float, float]]] = None
    oneplane_plane_tolerance: float = 0.02
    oneplane_face_normal_sign: float = -1.0
    oneplane_heading_yaw_offset_deg: float = 0.0
    oblique_oneplane_enabled: bool = False

    
    # Stage 4 transition parameters (Theta*)
    transition_grid_resolution: float = 2.0
    transition_max_expansions: int = 30000
    transition_goal_tolerance: float = 2.0
    transition_waypoint_spacing: float = 4.0
    transition_prefer_lateral_before_altitude: bool = True
    transition_lateral_offset_min_m: float = 6.0
    transition_lateral_offset_max_m: float = 30.0
    transition_lateral_offset_step_m: float = 4.0
    transition_lateral_max_candidates: int = 16
    transition_lateral_turn_penalty_weight: float = 0.15
    transition_enable_theta_star_fallback: bool = True

    # Coverage interpolation/capture sampling parameters
    capture_interpolation_factor: float = 0.5
    
    # Collision detection parameters
    voxel_size: float = 1.0  # Voxel grid resolution for collision detection (meters)
    collision_check_use_ray_casting: bool = False  # Enable ray casting for internal collision detection

    # Oblique-specific parameters
    oblique_dst_srf: float = 5.0  # Distance between parallel lines in oblique pattern (meters)
    oblique_min_altitude: float = 5.0  # Minimum altitude for oblique photography (meters)

    # Viewpoint-optimized layered ring sampling parameters
    viewpoint_layer_height_step_m: float = 2.5
    viewpoint_boundary_expand_m: float = 4.0
    viewpoint_ring_arc_step_m: float = 2.5
    viewpoint_min_points_per_layer: int = 8
    viewpoint_layer_order: str = "bottom_up"  # bottom_up|top_down
    viewpoint_min_altitude: float = 0.0  # meters above pointcloud min_z
    viewpoint_beyond_altitude: float = 0.0  # meters above pointcloud max_z
    viewpoint_shape_use_full_points: bool = False  # False=speed, True=boundary precision
    viewpoint_shape_roundness_m: float = 0.0  # 0=no rounding, >0 smoother shape corners
    viewpoint_shape_method: str = "alpha"  # alpha | convex
    viewpoint_hull_use_full_points: bool = False  # backward compatibility alias
    viewpoint_hull_roundness_m: float = 0.0  # backward compatibility alias
    viewpoint_alpha: float = 6.0  # alpha-shape radius threshold in meters (smaller = more concave)
    viewpoint_layer_area_jump_ratio: float = 1.6  # >1.0, trigger insertion when adjacent area ratio exceeds this
    viewpoint_layer_insert_max_global: int = 8  # cap total inserted interpolation layers
    
    def __post_init__(self):
        """Validate configuration."""
        if self.data_source_type not in ["auto", "pointcloud_file", "tileset"]:
            raise ValueError("data_source_type must be 'auto', 'pointcloud_file' or 'tileset'")

        scan_mode = str(getattr(self, "scan_direction_mode", "auto")).strip().lower()
        if scan_mode not in {"auto", "horizontal", "vertical", "swap"}:
            raise ValueError("scan_direction_mode must be 'auto', 'horizontal', 'vertical', or 'swap'")
        self.scan_direction_mode = scan_mode

        has_spiral_circle_params = any(
            value is not None
            for value in (
                self.spiral_center_xy,
                self.spiral_radius,
                self.spiral_start_z,
                self.spiral_height,
            )
        )
        if has_spiral_circle_params:
            if self.algorithm != "spiral":
                raise ValueError("spiral_* parameters are only supported when algorithm='spiral'")
            if (
                self.spiral_center_xy is None
                or self.spiral_radius is None
                or self.spiral_start_z is None
                or self.spiral_height is None
            ):
                raise ValueError(
                    "spiral_center_xy, spiral_radius, spiral_start_z, and spiral_height must all be provided together"
                )
            if len(self.spiral_center_xy) != 2:
                raise ValueError("spiral_center_xy must be a 2-tuple: (cx, cy)")
            if float(self.spiral_radius) <= 0:
                raise ValueError("spiral_radius must be > 0")
            if float(self.spiral_height) <= 0:
                raise ValueError("spiral_height must be > 0")
            self.spiral_circle_only_enabled = True

        has_cylinder_params = any(
            value is not None
            for value in (
                self.cylinder_center_xy,
                self.cylinder_radius,
                self.cylinder_start_z,
                self.cylinder_height,
            )
        )
        if has_cylinder_params:
            if self.algorithm != "cylinder":
                raise ValueError("cylinder_* parameters are only supported when algorithm='cylinder'")
            if (
                self.cylinder_center_xy is None
                or self.cylinder_radius is None
                or self.cylinder_start_z is None
                or self.cylinder_height is None
            ):
                raise ValueError(
                    "cylinder_center_xy, cylinder_radius, cylinder_start_z, and cylinder_height must all be provided together"
                )
            if len(self.cylinder_center_xy) != 2:
                raise ValueError("cylinder_center_xy must be a 2-tuple: (cx, cy)")
            if float(self.cylinder_radius) <= 0:
                raise ValueError("cylinder_radius must be > 0")
            if float(self.cylinder_height) <= 0:
                raise ValueError("cylinder_height must be > 0")
            self.cylinder_params_enabled = True

        if self.cylinder_mode not in ("horizontal", "vertical"):
            raise ValueError("cylinder_mode must be 'horizontal' or 'vertical'")

        has_oneplane_polygon = self.oneplane_polygon_xyz is not None
        if has_oneplane_polygon:
            if self.algorithm != "oblique_oneplane":
                raise ValueError("oneplane_polygon_xyz is only supported when algorithm='oblique_oneplane'")
            if len(self.oneplane_polygon_xyz) < 3:
                raise ValueError("oneplane_polygon_xyz must contain at least 3 points")
            polygon_points = np.asarray(self.oneplane_polygon_xyz, dtype=np.float64)
            if polygon_points.ndim != 2 or polygon_points.shape[1] != 3:
                raise ValueError("oneplane_polygon_xyz must be a list of 3D points")

            centroid = np.mean(polygon_points, axis=0)
            centered = polygon_points - centroid
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm < 1e-6:
                raise ValueError("oneplane_polygon_xyz points are collinear, cannot fit a plane")
            normal = normal / normal_norm
            distances = np.abs(centered @ normal)
            tolerance = float(getattr(self, "oneplane_plane_tolerance", 0.02))
            if float(np.max(distances)) > tolerance:
                raise ValueError("oneplane_polygon_xyz points are not coplanar within tolerance")

            sign = float(getattr(self, "oneplane_face_normal_sign", -1.0))
            if sign == 0.0:
                raise ValueError("oneplane_face_normal_sign must be non-zero")

            self.oblique_oneplane_enabled = True

        if self.coverage_area_rect_xy is not None:
            if len(self.coverage_area_rect_xy) != 4:
                raise ValueError("coverage_area_rect_xy must be a 4-tuple: (xmin, ymin, xmax, ymax)")
            xmin, ymin, xmax, ymax = [float(value) for value in self.coverage_area_rect_xy]
            if not (xmin < xmax and ymin < ymax):
                raise ValueError("coverage_area_rect_xy must satisfy xmin < xmax and ymin < ymax")

        if self.data_source_type == "auto":
            input_path = self.tileset_path or self.pointcloud_path
            if (
                not self.region_only_enabled
                and not self.spiral_circle_only_enabled
                and not self.cylinder_params_enabled
                and not self.oblique_oneplane_enabled
                and not input_path
                and self.algorithm == "boustrophedon"
                and self.coverage_area_rect_xy is not None
            ):
                self.region_only_enabled = True

        if self.region_only_enabled:
            if self.algorithm != "boustrophedon":
                raise ValueError("region_only_enabled currently supports only algorithm='boustrophedon'")
            if self.coverage_area_rect_xy is None:
                raise ValueError("coverage_area_rect_xy is required when region_only_enabled=True")

        if (
            not self.region_only_enabled
            and not self.spiral_circle_only_enabled
            and not self.cylinder_params_enabled
            and not self.oblique_oneplane_enabled
            and self.data_source_type == "auto"
        ):
            input_path = self.tileset_path or self.pointcloud_path
            # Handle empty strings (used in tests)
            input_path = input_path if input_path else None
            if not input_path:
                raise ValueError("Either pointcloud_path or tileset_path is required when data_source_type='auto'")

            suffix = Path(input_path).suffix.lower()
            if suffix == ".json":
                self.data_source_type = "tileset"
                if not self.tileset_path:
                    self.tileset_path = input_path
            elif suffix in [".pcd", ".ply"]:
                self.data_source_type = "pointcloud_file"
                if not self.pointcloud_path:
                    self.pointcloud_path = input_path
            else:
                raise ValueError(
                    "Cannot infer data source from file suffix. Use .json for tileset or .pcd/.ply for point cloud"
                )

        if self.region_only_enabled or self.spiral_circle_only_enabled or self.oblique_oneplane_enabled:
            pass
        elif self.cylinder_params_enabled:
            pass
        elif self.data_source_type == "pointcloud_file":
            if not self.pointcloud_path and self.tileset_path:
                self.pointcloud_path = self.tileset_path

            if not self.pointcloud_path:
                raise ValueError("pointcloud_path is required when data_source_type='pointcloud_file'")

            # Only validate file path if it's not a placeholder
            if self.pointcloud_path and self.pointcloud_path != "sample.pcd":
                if not Path(self.pointcloud_path).exists():
                    raise FileNotFoundError(
                        f"Point cloud file not found: {self.pointcloud_path}"
                    )
        else:
            if not self.tileset_path:
                if self.pointcloud_path and Path(self.pointcloud_path).suffix.lower() == ".json":
                    self.tileset_path = self.pointcloud_path
                else:
                    raise ValueError("tileset_path is required when data_source_type='tileset'")

            if not self.tileset_path:
                raise ValueError("tileset_path is required when data_source_type='tileset'")
            if not Path(self.tileset_path).exists():
                raise FileNotFoundError(f"3D Tiles file not found: {self.tileset_path}")

            if self.tiles_kind not in ["pointcloud", "model", "auto"]:
                raise ValueError("tiles_kind must be 'pointcloud', 'model', or 'auto'")

            if self.tiles_max_points <= 0:
                raise ValueError("tiles_max_points must be > 0")

            if self.tiles_lod_max is not None and self.tiles_lod_max < 0:
                raise ValueError("tiles_lod_max must be >= 0 when provided")

            if self.tiles_bbox is not None and len(self.tiles_bbox) != 6:
                raise ValueError("tiles_bbox must be a 6-tuple: (min_x,max_x,min_y,max_y,min_z,max_z)")

            if self.tiles_output_frame not in ["world", "enu"]:
                raise ValueError("tiles_output_frame must be 'world' or 'enu'")

            if self.tiles_input_crs not in ["auto", "ecef"]:
                raise ValueError("tiles_input_crs must be 'auto' or 'ecef'")

            if self.tiles_enu_origin_ecef is not None and len(self.tiles_enu_origin_ecef) != 3:
                raise ValueError("tiles_enu_origin_ecef must be a 3-tuple: (x,y,z)")

            if self.tiles_convert_coord_frame not in ["world", "enu", "centroid", "first_point"]:
                raise ValueError(
                    "tiles_convert_coord_frame must be one of: 'world', 'enu', 'centroid', 'first_point'"
                )

            if self.tiles_convert_to_ply and self.tiles_converted_ply_path is not None:
                if Path(self.tiles_converted_ply_path).suffix.lower() != ".ply":
                    raise ValueError("tiles_converted_ply_path must end with .ply")
        
        valid_algorithms = [
            "boustrophedon",
            "spiral",
            "cylinder",
            "oblique",
            "oblique_oneplane",
            "viewpoint_optimized",
            "viewpoint_wrap",
        ]
        if self.algorithm not in valid_algorithms:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm}. "
                f"Valid options: {valid_algorithms}"
            )
        
        if not 0 <= self.coverage_threshold <= 1:
            raise ValueError("coverage_threshold must be between 0 and 1")
        
        if not 0 <= self.side_overlap <= 1:
            raise ValueError("side_overlap must be between 0 and 1")
        
        if not 0 <= self.front_overlap <= 1:
            raise ValueError("front_overlap must be between 0 and 1")

        if self.gsd is not None and self.gsd <= 0:
            raise ValueError("gsd must be > 0 when provided")

        if self.aabb_center is not None and len(self.aabb_center) != 3:
            raise ValueError("aabb_center must be a 3-tuple: (cx, cy, cz)")

        if self.aabb_size is not None:
            if len(self.aabb_size) != 3:
                raise ValueError("aabb_size must be a 3-tuple: (sx, sy, sz)")
            if any(float(value) <= 0 for value in self.aabb_size):
                raise ValueError("aabb_size values must all be > 0")
            if self.aabb_transform is None and self.aabb_center is None:
                raise ValueError("aabb_center is required when aabb_size is set and aabb_transform is not provided")

        if self.aabb_transform is not None:
            matrix = np.asarray(self.aabb_transform, dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError("aabb_transform must be a 4x4 matrix")

        if self.global_distance_m is not None and self.global_distance_m <= 0:
            raise ValueError("global_distance_m must be > 0 when provided")

        if not isinstance(self.region_ground_z, (int, float)):
            raise ValueError("region_ground_z must be numeric")

        # Backward-compatible replacement: old hull params -> new shape params
        if (not bool(self.viewpoint_shape_use_full_points)) and bool(self.viewpoint_hull_use_full_points):
            self.viewpoint_shape_use_full_points = True
        if float(self.viewpoint_shape_roundness_m) == 0.0 and float(self.viewpoint_hull_roundness_m) != 0.0:
            self.viewpoint_shape_roundness_m = float(self.viewpoint_hull_roundness_m)

        if self.waypoint_export_frame not in ["planning", "source"]:
            raise ValueError("waypoint_export_frame must be 'planning' or 'source'")

        if self.min_flight_altitude_m is not None and self.min_flight_altitude_m <= 0:
            raise ValueError("min_flight_altitude_m must be > 0 when provided")

        # GSD as precision requirement:
        # - If current global_distance_m already satisfies target GSD, keep it.
        # - Otherwise, tighten distance using camera-model conversion.
        # - If global_distance_m is missing, derive one from GSD.
        if self.gsd is not None:
            target_gsd = float(self.gsd)
            if self.global_distance_m is None:
                self.global_distance_m = float(gsd_to_global_distance_m(self.camera, target_gsd))
                print(
                    f"[MissionConfig] gsd requirement={target_gsd:.6f} m/px; "
                    f"global_distance_m not set, derived={self.global_distance_m:.3f} m"
                )
            else:
                current_distance = float(self.global_distance_m)
                current_gsd = float(oblique_dst_srf_to_gsd(self.camera, current_distance))

                if current_gsd <= target_gsd:
                    print(
                        f"[MissionConfig] gsd requirement={target_gsd:.6f} m/px; "
                        f"current global_distance_m={current_distance:.3f} m already meets it "
                        f"(current_gsd={current_gsd:.6f} m/px), keeping current distance"
                    )
                else:
                    required_distance = float(gsd_to_global_distance_m(self.camera, target_gsd))
                    self.global_distance_m = required_distance
                    print(
                        f"[MissionConfig] gsd requirement={target_gsd:.6f} m/px; "
                        f"current global_distance_m={current_distance:.3f} m gives "
                        f"current_gsd={current_gsd:.6f} m/px (too coarse), "
                        f"updated global_distance_m={required_distance:.3f} m"
                    )

        if not 0.05 <= self.capture_interpolation_factor <= 3.0:
            raise ValueError("capture_interpolation_factor must be between 0.05 and 3.0")

        if self.viewpoint_layer_height_step_m <= 0:
            raise ValueError("viewpoint_layer_height_step_m must be > 0")

        if self.viewpoint_boundary_expand_m < 0:
            raise ValueError("viewpoint_boundary_expand_m must be >= 0")

        if self.viewpoint_ring_arc_step_m <= 0:
            raise ValueError("viewpoint_ring_arc_step_m must be > 0")

        if self.viewpoint_min_points_per_layer < 3:
            raise ValueError("viewpoint_min_points_per_layer must be >= 3")

        if self.viewpoint_layer_order not in ["bottom_up", "top_down"]:
            raise ValueError("viewpoint_layer_order must be 'bottom_up' or 'top_down'")

        if self.viewpoint_min_altitude < 0:
            raise ValueError("viewpoint_min_altitude must be >= 0")

        if self.viewpoint_beyond_altitude < 0:
            raise ValueError("viewpoint_beyond_altitude must be >= 0")

        if self.viewpoint_shape_roundness_m < 0:
            raise ValueError("viewpoint_shape_roundness_m must be >= 0")

        if self.viewpoint_shape_method not in ["alpha", "convex"]:
            raise ValueError("viewpoint_shape_method must be 'alpha' or 'convex'")

        if self.viewpoint_hull_roundness_m < 0:
            raise ValueError("viewpoint_hull_roundness_m must be >= 0")

        if self.viewpoint_alpha <= 0:
            raise ValueError("viewpoint_alpha must be > 0")

        if self.viewpoint_layer_area_jump_ratio <= 1.0:
            raise ValueError("viewpoint_layer_area_jump_ratio must be > 1.0")

        if self.viewpoint_layer_insert_max_global < 0:
            raise ValueError("viewpoint_layer_insert_max_global must be >= 0")

        if self.transition_lateral_offset_min_m <= 0:
            raise ValueError("transition_lateral_offset_min_m must be > 0")

        if self.transition_lateral_offset_max_m <= 0:
            raise ValueError("transition_lateral_offset_max_m must be > 0")

        if self.transition_lateral_offset_max_m < self.transition_lateral_offset_min_m:
            raise ValueError("transition_lateral_offset_max_m must be >= transition_lateral_offset_min_m")

        if self.transition_lateral_offset_step_m <= 0:
            raise ValueError("transition_lateral_offset_step_m must be > 0")

        if self.transition_lateral_max_candidates < 2:
            raise ValueError("transition_lateral_max_candidates must be >= 2")

        if self.transition_lateral_turn_penalty_weight < 0:
            raise ValueError("transition_lateral_turn_penalty_weight must be >= 0")

