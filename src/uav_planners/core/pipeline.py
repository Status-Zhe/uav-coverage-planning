"""Four-stage coverage planning pipeline."""

from typing import List, Union, Tuple, Optional
import numpy as np
import logging
from pathlib import Path
from scipy.spatial import cKDTree

from .mission_config import MissionConfig
from .planner_result import PlannerResult
from .data_loader import DataLoader
from ..geometry.base_generator import GeneratorConfig
from ..geometry.registry import get_generator
from ..geometry.generators import *
from ..geometry.generators import TransitionThetaStarGenerator
from ..constraints.collision_checker import CollisionChecker
from ..optimization.tsp_solver import TSPSolver
from ..optimization.route_sequence_optimizer import RouteSequenceOptimizer
from ..optimization.capture_point_sampler import CapturePointSampler
from ..models.pointcloud import PointCloud
from ..models.waypoint import Waypoint, WaypointAction
from ..models.route_collection import RouteCollection, RouteMetadata


logger = logging.getLogger(__name__)


class CoveragePipeline:
    """Four-stage coverage planning pipeline.
    
    Pipeline stages:
    1. Geometry Generation: Generate raw waypoints using selected algorithm
    2. Constraint Validation: Filter waypoints for safety/collision
    3. Optimization: TSP ordering
    4. Transition + adaptive interpolation and capture sampling
    
    Example:
        config = MissionConfig(
            pointcloud_path="building.pcd",
            camera=camera,
            algorithm="boustrophedon"
        )
        pipeline = CoveragePipeline(config)
        result = pipeline.run(pointcloud)
    """
    
    def __init__(self, config: MissionConfig):
        """Initialize pipeline with configuration.
        
        Args:
            config: Mission configuration
        """
        self.config = config
        # Cache for cKDTree to avoid redundant construction overhead
        self._target_pointcloud_tree: Optional[cKDTree] = None
        self._obstacle_pointcloud_tree: Optional[cKDTree] = None
        self._init_stage1()
        self._init_stage2()
        self._init_stage3()
        self._init_stage4()
    
    def _init_stage1(self):
        """Initialize Stage 1: Geometry Generator."""
        generator_class = get_generator(self.config.algorithm)
        self.generator = generator_class()
        self.gen_config = GeneratorConfig(
            altitude=self.config.altitude,
            speed_ms=self.config.speed_ms,
            side_overlap=self.config.side_overlap,
            front_overlap=self.config.front_overlap,
            scan_direction_mode=getattr(self.config, "scan_direction_mode", "auto"),
            safety_distance=self.config.safety_distance,
            coverage_threshold=self.config.coverage_threshold,
            global_distance_m=self.config.global_distance_m,
            min_flight_altitude_m=self.config.min_flight_altitude_m,
            region_only_enabled=bool(getattr(self.config, "region_only_enabled", False)),
            coverage_area_rect_xy=getattr(self.config, "coverage_area_rect_xy", None),
            region_ground_z=float(getattr(self.config, "region_ground_z", 0.0)),
            oblique_dst_srf=self.config.oblique_dst_srf,
            oblique_min_altitude=self.config.oblique_min_altitude,
            viewpoint_layer_height_step_m=self.config.viewpoint_layer_height_step_m,
            viewpoint_boundary_expand_m=self.config.viewpoint_boundary_expand_m,
            viewpoint_ring_arc_step_m=self.config.viewpoint_ring_arc_step_m,
            viewpoint_min_points_per_layer=self.config.viewpoint_min_points_per_layer,
            viewpoint_layer_order=self.config.viewpoint_layer_order,
            viewpoint_min_altitude=self.config.viewpoint_min_altitude,
            viewpoint_beyond_altitude=self.config.viewpoint_beyond_altitude,
            viewpoint_shape_use_full_points=self.config.viewpoint_shape_use_full_points,
            viewpoint_shape_roundness_m=self.config.viewpoint_shape_roundness_m,
            viewpoint_shape_method=self.config.viewpoint_shape_method,
            viewpoint_hull_use_full_points=self.config.viewpoint_hull_use_full_points,
            viewpoint_hull_roundness_m=self.config.viewpoint_hull_roundness_m,
            viewpoint_alpha=self.config.viewpoint_alpha,
            viewpoint_layer_area_jump_ratio=self.config.viewpoint_layer_area_jump_ratio,
            viewpoint_layer_insert_max_global=self.config.viewpoint_layer_insert_max_global,
            transition_prefer_lateral_before_altitude=self.config.transition_prefer_lateral_before_altitude,
            transition_lateral_offset_min_m=self.config.transition_lateral_offset_min_m,
            transition_lateral_offset_max_m=self.config.transition_lateral_offset_max_m,
            transition_lateral_offset_step_m=self.config.transition_lateral_offset_step_m,
            transition_lateral_max_candidates=self.config.transition_lateral_max_candidates,
            transition_lateral_turn_penalty_weight=self.config.transition_lateral_turn_penalty_weight,
            transition_enable_theta_star_fallback=self.config.transition_enable_theta_star_fallback,
            spiral_center_xy=getattr(self.config, "spiral_center_xy", None),
            spiral_radius=getattr(self.config, "spiral_radius", None),
            spiral_start_z=getattr(self.config, "spiral_start_z", None),
            spiral_height=getattr(self.config, "spiral_height", None),
            cylinder_center_xy=getattr(self.config, "cylinder_center_xy", None),
            cylinder_radius=getattr(self.config, "cylinder_radius", None),
            cylinder_start_z=getattr(self.config, "cylinder_start_z", None),
            cylinder_height=getattr(self.config, "cylinder_height", None),
            cylinder_mode=getattr(self.config, "cylinder_mode", None),
            cylinder_ring_spacing_m=getattr(self.config, "cylinder_ring_spacing_m", None),
            cylinder_ring_count=getattr(self.config, "cylinder_ring_count", None),
            cylinder_strip_spacing_m=getattr(self.config, "cylinder_strip_spacing_m", None),
            cylinder_strip_count=getattr(self.config, "cylinder_strip_count", None),
            cylinder_angle_start_deg=getattr(self.config, "cylinder_angle_start_deg", None),
            oneplane_polygon_xyz=getattr(self.config, "oneplane_polygon_xyz", None),
            oneplane_plane_tolerance=getattr(self.config, "oneplane_plane_tolerance", None),
            oneplane_face_normal_sign=getattr(self.config, "oneplane_face_normal_sign", None),
            oneplane_heading_yaw_offset_deg=getattr(self.config, "oneplane_heading_yaw_offset_deg", None),
        )
        
        logger.info(f"Stage 1: Initialized {self.generator.name} generator")
    
    def _init_stage2(self):
        """Initialize Stage 2: Constraint Validators."""
        # Collision checker initialized during run (needs pointcloud)
        self.collision_checker = None
        logger.info("Stage 2: Constraint validators ready")
    
    def _init_stage3(self):
        """Initialize Stage 3: Optimizers."""
        self.tsp_solver = TSPSolver(method="auto")
        self.route_sequence_optimizer = RouteSequenceOptimizer(method="greedy")
        self.capture_sampler = CapturePointSampler()
        logger.info("Stage 3: Optimizers initialized")
    
    def _init_stage4(self):
        """Initialize Stage 4: Transition Path Generator.
        
        Note: TransitionThetaStarGenerator is instantiated per-transition in
        _stage4_generate_transitions for better configurability.
        """
        logger.info("Stage 4: Theta* transition generator ready")
    
    def run(self, target_pointcloud: PointCloud, obstacle_pointcloud: Optional[PointCloud] = None) -> PlannerResult:
        """Execute the three-stage pipeline.
        
        Args:
            target_pointcloud: Target structure point cloud for waypoint generation
            obstacle_pointcloud: Environment point cloud for collision/transition checks.
                If None, target_pointcloud is used.
            
        Returns:
            Planning result with trajectory and coverage info
        """
        if obstacle_pointcloud is None:
            obstacle_pointcloud = target_pointcloud

        # Cache cKDTree once at pipeline start to avoid redundant construction overhead
        self._target_pointcloud_tree = cKDTree(target_pointcloud.points) if target_pointcloud.point_count > 0 else None
        self._obstacle_pointcloud_tree = cKDTree(obstacle_pointcloud.points) if obstacle_pointcloud.point_count > 0 else None
        
        # Store obstacle cloud for stage 4 access
        self.pointcloud = obstacle_pointcloud
        
        # Stage 1: Geometry Generation
        raw_waypoints = self._stage1_generate(target_pointcloud)
        if not raw_waypoints:
            logger.warning("No waypoints generated")
            return PlannerResult(waypoints=[])
        
        # Convert to RouteCollection if needed (handles both single route and multiple routes)
        if isinstance(raw_waypoints, list) and raw_waypoints and isinstance(raw_waypoints[0], list):
            # Multiple routes returned from generator
            raw_routes = RouteCollection(raw_waypoints)
        else:
            # Single route - wrap in RouteCollection
            raw_routes = RouteCollection([raw_waypoints], ["default"])
        
        # Stage 2: Constraint Validation
        if bool(getattr(self.config, "region_only_enabled", False)) or bool(
            getattr(self.config, "oblique_oneplane_enabled", False)
        ):
            logger.info("Stage 2: Skipped collision validation in region-only mode")
            safe_routes = raw_routes
        else:
            safe_routes = self._stage2_validate(raw_routes, obstacle_pointcloud)
        if not safe_routes.routes or all(len(r) == 0 for r in safe_routes.routes):
            logger.warning("No safe waypoints after validation")
            return PlannerResult(waypoints=[])
        
        # Stage 3: Optimization
        optimized_routes = self._stage3_optimize(safe_routes)
        
        # Stage 4: Generate transition paths between routes
        transitions = self._stage4_generate_transitions(optimized_routes)
        
        # Build continuous path: routes + transitions interleaved
        continuous_waypoints = []
        for i, route in enumerate(optimized_routes.routes):
            for wp in route:
                if getattr(wp, "waypoint_type", None) is None:
                    wp.waypoint_type = "coverage"
                wp.parent_route_id = i
            continuous_waypoints.extend(route)
            if i < len(transitions) and transitions[i]:
                # Skip first waypoint of transition (same as route exit)
                for wp in transitions[i][1:]:
                    wp.waypoint_type = "transition"
                    wp.parent_route_id = i
                continuous_waypoints.extend(transitions[i][1:])
        
        # Stage 5: Adaptive interpolation and coverage capture sampling
        continuous_waypoints = self._postprocess_trajectory_for_capture(continuous_waypoints)

        # Stage 6: Final collision re-check on full trajectory
        continuous_waypoints = self._stage6_final_collision_check(continuous_waypoints)

        # Build result
        safe_flattened = safe_routes.flattened

        planning_frame = str(getattr(obstacle_pointcloud, "coordinate_frame", "world"))
        enu_origin_ecef = getattr(obstacle_pointcloud, "enu_origin_ecef", None)
        source_frame = "world"
        if self.config.data_source_type == "tileset":
            if str(getattr(self.config, "tiles_output_frame", "enu")).lower() == "enu":
                if str(getattr(self.config, "tiles_input_crs", "auto")).lower() == "ecef":
                    source_frame = "ecef"
                else:
                    source_frame = "unknown"
            else:
                source_frame = "world"
        
        # Save transitions to metadata for export
        transitions_data = []
        for i, trans in enumerate(transitions):
            if trans:
                transitions_data.append({
                    "from_route": i,
                    "to_route": i + 1,
                    "waypoint_count": len(trans),
                    "waypoints": [
                        {
                            "x": wp.x, "y": wp.y, "z": wp.z,
                            "heading_deg": wp.heading_deg,
                            "is_keypoint": wp.is_keypoint,
                            "waypoint_type": getattr(wp, 'waypoint_type', 'transition'),
                            "parent_route_id": getattr(wp, 'parent_route_id', None),
                        } for wp in trans
                    ]
                })
        
        result = PlannerResult(
            waypoints=continuous_waypoints,
            sparse_waypoints=safe_flattened,  # P0: all are keypoints
            dense_trajectory=continuous_waypoints,  # P0: no interpolation
            routes=optimized_routes,  # Preserve multi-route structure
            metadata={
                "algorithm": self.config.algorithm,
                "altitude": self.config.altitude,
                "planning_frame": planning_frame,
                "source_frame": source_frame,
                "enu_origin_ecef": enu_origin_ecef,
                "waypoint_export_frame": getattr(self.config, "waypoint_export_frame", "planning"),
                "raw_routes": raw_routes.route_count,
                "safe_routes": safe_routes.route_count,
                "final_routes": optimized_routes.route_count,
                "raw_waypoints": len(raw_routes.flattened),
                "safe_waypoints": len(safe_flattened),
                "final_waypoints": len(continuous_waypoints),
                "capture_points": len([wp for wp in continuous_waypoints if wp.action == WaypointAction.SHOOT]),
                "transition_paths": len([t for t in transitions if t]),
                "transitions": transitions_data,
            }
        )
        
        logger.info(
            f"Pipeline complete: {raw_routes.route_count} routes, "
            f"{len(raw_routes.flattened)} raw -> {len(safe_flattened)} safe -> "
            f"{len(continuous_waypoints)} final with transitions"
        )
        
        return result

    def _stage6_final_collision_check(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Final safety check after transitions/interpolation.

        Ensures newly introduced transition/interpolated waypoints are also
        collision-free against the obstacle point cloud used in Stage 2.
        """
        if bool(getattr(self.config, "region_only_enabled", False)) or bool(
            getattr(self.config, "oblique_oneplane_enabled", False)
        ):
            logger.info("Stage 6: Skipped final collision check in region-only mode")
            return waypoints

        if not waypoints or self.collision_checker is None:
            return waypoints

        # print("Stage 6: Performing final collision re-check on full trajectory")  # Debug log for stage 6 start --- IGNORE ---
        validation = self.collision_checker.check_trajectory(waypoints)
        if validation.valid:
            # logger.info("Stage 6: Final collision re-check passed (%d waypoints)", len(waypoints))
            return waypoints

        safe_indices = []
        for index, waypoint in enumerate(waypoints):
            position = np.array([waypoint.x, waypoint.y, waypoint.z], dtype=float)
            if not self.collision_checker.check_collision(position):
                safe_indices.append(index)

        if not safe_indices:
            logger.warning(
                # "Stage 6: Final collision re-check removed all waypoints (%d -> 0)",
                len(waypoints),
            )
            return []

        # collision_removed = len(waypoints) - len(safe_indices)

        # Important order in Stage 6:
        # 1) Remove colliding waypoints
        # 2) Remove bad keypoints by missing-ratio rule
        # 3) Repair the remaining gaps in one pass
        refined_safe_indices = self._stage6_prune_bad_keypoints(
            original_waypoints=waypoints,
            safe_indices=safe_indices,
            missing_ratio_threshold=0.75,
        )

        if not refined_safe_indices:
            logger.warning("Stage 6: No waypoints left after keypoint pruning")
            return []

        repaired_waypoints = self._repair_stage6_gaps_once(waypoints, refined_safe_indices)
        return repaired_waypoints

    def _stage6_prune_bad_keypoints(
        self,
        original_waypoints: List[Waypoint],
        safe_indices: List[int],
        missing_ratio_threshold: float = 0.75,
    ) -> List[int]:
        """Drop interior keypoints whose adjacent key-segments lose too many midpoints.

        For each segment between consecutive keypoints, if the ratio of missing
        intermediate points is above threshold, mark segment endpoint keypoints
        (excluding global start/end keypoints) for removal.
        """
        if len(safe_indices) < 3:
            return safe_indices

        keypoint_indices = [
            index for index, waypoint in enumerate(original_waypoints)
            if bool(getattr(waypoint, "is_keypoint", False))
        ]
        if len(keypoint_indices) < 3:
            return safe_indices

        safe_index_set = set(safe_indices)
        keypoints_to_drop = set()

        for segment_idx in range(len(keypoint_indices) - 1):
            left_key_idx = keypoint_indices[segment_idx]
            right_key_idx = keypoint_indices[segment_idx + 1]

            total_inner = max(0, right_key_idx - left_key_idx - 1)
            if total_inner <= 0:
                continue

            safe_inner = 0
            for point_idx in range(left_key_idx + 1, right_key_idx):
                if point_idx in safe_index_set:
                    safe_inner += 1

            missing_ratio = (total_inner - safe_inner) / total_inner
            if missing_ratio < missing_ratio_threshold:
                continue

            if segment_idx > 0 and left_key_idx in safe_index_set:
                keypoints_to_drop.add(left_key_idx)
            if segment_idx + 1 < len(keypoint_indices) - 1 and right_key_idx in safe_index_set:
                keypoints_to_drop.add(right_key_idx)

        if not keypoints_to_drop:
            return safe_indices

        return [index for index in safe_indices if index not in keypoints_to_drop]

    def _repair_stage6_gaps_once(self, original_waypoints: List[Waypoint], safe_indices: List[int]) -> List[Waypoint]:
        """Repair disconnected segments after one-pass collision filtering.

        This method is intentionally called once after all colliding waypoints
        are removed, to avoid repeated expensive transition generation.
        """
        if not safe_indices:
            return []
        if len(safe_indices) == 1:
            return [original_waypoints[safe_indices[0]]]

        stage6_max_expansions = int(
            getattr(
                self.config,
                "stage6_transition_max_expansions",
                max(3000, int(self.config.transition_max_expansions) // 5),
            )
        )
        stage6_enable_theta = bool(getattr(self.config, "transition_enable_theta_star_fallback", False))

        generator = TransitionThetaStarGenerator(
            grid_resolution=self.config.transition_grid_resolution,
            max_expansions=stage6_max_expansions,
            safety_margin=self.config.safety_distance,
            goal_tolerance=self.config.transition_goal_tolerance,
            waypoint_spacing=self.config.transition_waypoint_spacing,
        )

        repaired: List[Waypoint] = [original_waypoints[safe_indices[0]]]
        repaired_segment_count = 0
        attempted_repairs = 0
        skipped_by_budget = 0

        max_repair_segments = max(0, int(getattr(self.config, "stage6_max_repair_segments", 24)))
        max_repair_gap_m = max(0.0, float(getattr(self.config, "stage6_max_repair_gap_m", 80.0)))

        for safe_pos in range(1, len(safe_indices)):
            left_original_index = safe_indices[safe_pos - 1]
            right_original_index = safe_indices[safe_pos]

            left_waypoint = original_waypoints[left_original_index]
            right_waypoint = original_waypoints[right_original_index]

            if right_original_index == left_original_index + 1:
                repaired.append(right_waypoint)
                continue

            if attempted_repairs >= max_repair_segments:
                skipped_by_budget += 1
                repaired.append(right_waypoint)
                continue

            gap_distance = float(
                np.linalg.norm(
                    np.array([right_waypoint.x - left_waypoint.x, right_waypoint.y - left_waypoint.y, right_waypoint.z - left_waypoint.z], dtype=float)
                )
            )
            if gap_distance > max_repair_gap_m:
                skipped_by_budget += 1
                repaired.append(right_waypoint)
                continue

            attempted_repairs += 1

            bridge_config = GeneratorConfig(
                start_waypoint=left_waypoint,
                goal_waypoint=right_waypoint,
                transition_grid_resolution=self.config.transition_grid_resolution,
                transition_max_expansions=stage6_max_expansions,
                transition_goal_tolerance=self.config.transition_goal_tolerance,
                transition_waypoint_spacing=self.config.transition_waypoint_spacing,
                transition_prefer_lateral_before_altitude=self.config.transition_prefer_lateral_before_altitude,
                transition_lateral_offset_min_m=self.config.transition_lateral_offset_min_m,
                transition_lateral_offset_max_m=self.config.transition_lateral_offset_max_m,
                transition_lateral_offset_step_m=self.config.transition_lateral_offset_step_m,
                transition_lateral_max_candidates=self.config.transition_lateral_max_candidates,
                transition_lateral_turn_penalty_weight=self.config.transition_lateral_turn_penalty_weight,
                transition_enable_theta_star_fallback=stage6_enable_theta,
            )

            try:
                # Pass cached obstacle tree to avoid redundant cKDTree construction
                bridge = generator.generate(self.pointcloud, self.config.camera, bridge_config, self._obstacle_pointcloud_tree)[0]
            except Exception as exc:
                logger.warning(
                    "Stage 6: Failed to repair gap %d->%d with transition planner (%s), keeping direct connection",
                    left_original_index,
                    right_original_index,
                    exc,
                )
                repaired.append(right_waypoint)
                continue

            if not bridge:
                repaired.append(right_waypoint)
                continue

            repaired_segment_count += 1
            bridge_tail = bridge[1:]
            # Set bridge waypoints to transition type with SHOOT action
            for waypoint in bridge_tail[:-1]:
                waypoint.waypoint_type = "transition"
                waypoint.action = WaypointAction.SHOOT
                waypoint.is_keypoint = False
                waypoint.parent_route_id = left_waypoint.parent_route_id
                self._stage6_orient_waypoint_to_pointcloud_normal(waypoint)

            repaired.extend(bridge_tail)

        if repaired_segment_count > 0:
            logger.info(
                "Stage 6: Repaired %d disconnected segments with one-pass transition stitching",
                repaired_segment_count,
            )

        if skipped_by_budget > 0:
            logger.info(
                "Stage 6: Skipped %d gap repairs due to stage6 budget/threshold limits",
                skipped_by_budget,
            )

        return repaired

    def _stage6_get_pointcloud_tree(self) -> Optional[cKDTree]:
        if not hasattr(self, "_stage6_point_tree"):
            self._stage6_point_tree = None
            if getattr(self, "pointcloud", None) is not None and self.pointcloud.point_count > 0:
                self._stage6_point_tree = cKDTree(self.pointcloud.points)
        return self._stage6_point_tree

    def _stage6_orient_waypoint_to_pointcloud_normal(self, waypoint: Waypoint) -> None:
        """Orient repaired bridge waypoint by updating aircraft heading only.

        Keeps gimbal pitch unchanged; only yaw/heading is aligned to nearest
        point-cloud normal (or nearest surface direction fallback).
        """
        tree = self._stage6_get_pointcloud_tree()
        if tree is None:
            return

        query = np.array([waypoint.x, waypoint.y, waypoint.z], dtype=float)
        _, nearest_idx = tree.query(query, k=1)
        nearest_idx = int(nearest_idx)

        target_point = self.pointcloud.points[nearest_idx]
        to_surface = target_point - query
        to_surface_norm = float(np.linalg.norm(to_surface))
        if to_surface_norm < 1e-6:
            return

        look_vec = to_surface / to_surface_norm
        normals = getattr(self.pointcloud, "normals", None)
        if normals is not None and nearest_idx < len(normals):
            normal = np.asarray(normals[nearest_idx], dtype=float)
            normal_norm = float(np.linalg.norm(normal))
            if np.isfinite(normal_norm) and normal_norm > 1e-6:
                normal = normal / normal_norm
                if float(np.dot(normal, to_surface)) < 0.0:
                    normal = -normal
                look_vec = normal

        to_surface_xy = np.array([to_surface[0], to_surface[1]], dtype=float)
        to_surface_xy_norm = float(np.linalg.norm(to_surface_xy))
        look_xy = np.array([look_vec[0], look_vec[1]], dtype=float)
        look_xy_norm = float(np.linalg.norm(look_xy))

        if look_xy_norm < 1e-6:
            if to_surface_xy_norm < 1e-6:
                return
            heading_xy = to_surface_xy / to_surface_xy_norm
        else:
            heading_xy = look_xy / look_xy_norm
            if to_surface_xy_norm >= 1e-6:
                inward_xy = to_surface_xy / to_surface_xy_norm
                if float(np.dot(heading_xy, inward_xy)) < 0.0:
                    heading_xy = -heading_xy

        heading = float(np.degrees(np.arctan2(heading_xy[0], heading_xy[1])))
        waypoint.heading_deg = heading
    
    def _stage1_generate(self, pointcloud: PointCloud) -> Union[List[Waypoint], List[List[Waypoint]]]:
        """Stage 1: Generate raw waypoints.
        
        Args:
            pointcloud: Target point cloud
            
        Returns:
            Raw waypoints from generator, either a single list or a list of lists
        """
        logger.info(f"Stage 1: Generating waypoints with {self.generator.name}")
        
        waypoints = self.generator.generate(
            pointcloud,
            self.config.camera,
            self.gen_config
        )
        if not waypoints:
            return []
        if isinstance(waypoints[0], list):
            logger.info(f"Stage 1: Generated {len(waypoints)} waypoints")
        
        return waypoints
    
    def _stage2_validate(
        self,
        routes: Union[List[Waypoint], RouteCollection],
        pointcloud: PointCloud
    ) -> RouteCollection:
        """Stage 2: Validate each route independently.
        
        Args:
            routes: Single route or RouteCollection with multiple routes
            pointcloud: Environment point cloud for collision checking
            
        Returns:
            RouteCollection with all routes validated
        """
        # Convert single route to RouteCollection
        if isinstance(routes, list):
            routes = RouteCollection([routes], ["default"])
        
        logger.info(f"Stage 2: Validating {routes.route_count} routes")
        
        # Initialize collision checker
        self.collision_checker = CollisionChecker(
            pointcloud,
            voxel_size=getattr(self.config, 'voxel_size', 1.0),
            safety_margin=self.config.safety_distance,
            use_ray_casting=bool(getattr(self.config, 'collision_check_use_ray_casting', False))
        )
        
        validated_routes = []
        for i, route in enumerate(routes.routes):
            label = routes.labels[i] if routes.labels else f"route_{i}"
            logger.info(f"  Validating route '{label}' with {len(route)} waypoints")
            
            safe_waypoints = self.collision_checker.filter_safe_waypoints(route)
            
            # Update metadata
            if routes.metadata and i < len(routes.metadata):
                routes.metadata[i].entry_waypoint = safe_waypoints[0] if safe_waypoints else None
                routes.metadata[i].exit_waypoint = safe_waypoints[-1] if safe_waypoints else None
            
            validated_routes.append(safe_waypoints)
            removed = len(route) - len(safe_waypoints)
            if removed > 0:
                logger.info(f"  Route '{label}': {removed} waypoints removed")
        
        result = RouteCollection(validated_routes, routes.labels, routes.metadata)
        total_waypoints = sum(len(r) for r in validated_routes)
        logger.info(f"Stage 2: Validation complete, {total_waypoints} waypoints total")
        return result
    
    def _stage3_optimize(
        self,
        route_collection: RouteCollection
    ) -> RouteCollection:
        """Stage 3: Optimize route sequence and internal waypoints.
        
        Two-level optimization:
        1. Route sequence: Use TSP to find optimal order of routes
           (minimizes transition distance between routes)
        2. Internal waypoints: Optimize waypoint order within each route
        
        Args:
            route_collection: Collection of routes to optimize
            
        Returns:
            Optimized RouteCollection with reordered routes
        """
        logger.info(f"Stage 3: Optimizing {route_collection.route_count} routes")
        
        # Step 1: Optimize route execution order using TSP
        # For layered viewpoint routes, preserve generator layer order.
        if self.config.algorithm in {"viewpoint_optimized", "viewpoint_wrap"}:
            ordered_collection = route_collection
        else:
            ordered_collection = self.route_sequence_optimizer.optimize(route_collection)
        
        # Step 2: Optimize internal waypoints for each route
        # Use TSP solver to optimize waypoint order within each route
        # optimized_routes = []
        # for route in ordered_collection.routes:
        #     if self.config.algorithm in {"viewpoint_optimized", "viewpoint_wrap"}:
        #         optimized_routes.append(route)
        #         continue
        #     if len(route) > 2:
        #         # Optimize waypoint sequence within this route
        #         optimized_route = self.tsp_solver.solve(route)
        #         optimized_routes.append(optimized_route)
        #     else:
        #         # Too few waypoints to optimize
        #         optimized_routes.append(route)
        
        if self.config.algorithm in {"viewpoint_optimized", "viewpoint_wrap"}:
            ordered_collection.routes = self._align_viewpoint_layer_route_entries(ordered_collection.routes)

        # Create result collection with optimized routes but preserving metadata
        result = RouteCollection(
            routes=ordered_collection.routes,
            labels=ordered_collection.labels,
            metadata=ordered_collection.metadata
        )
        
        logger.info(
            f"Stage 3: Optimized {result.route_count} routes, "
            f"{sum(len(r) for r in result.routes)} total waypoints"
        )
        return result

    def _align_viewpoint_layer_route_entries(self, routes: List[List[Waypoint]]) -> List[List[Waypoint]]:
        """Rotate each subsequent ring route to start near previous route's exit."""
        if not routes:
            return routes

        aligned = [routes[0]]
        for route in routes[1:]:
            if not route:
                aligned.append(route)
                continue

            prev_exit = aligned[-1][-1]
            prev_pos = np.array([prev_exit.x, prev_exit.y, prev_exit.z], dtype=float)

            best_index = 0
            best_dist = float("inf")
            for idx, waypoint in enumerate(route):
                candidate_pos = np.array([waypoint.x, waypoint.y, waypoint.z], dtype=float)
                dist = float(np.linalg.norm(candidate_pos - prev_pos))
                if dist < best_dist:
                    best_dist = dist
                    best_index = idx

            aligned.append(route[best_index:] + route[:best_index])

        return aligned
    
    def _stage4_generate_transitions(self, routes: RouteCollection) -> List[List[Waypoint]]:
        """Generate transitions using deterministic Theta* transition planner."""
        if bool(getattr(self.config, "oblique_oneplane_enabled", False)):
            logger.info("Stage 4: Skipped transitions in oblique one-plane mode")
            return []

        if bool(getattr(self.config, "region_only_enabled", False)):
            logger.info(f"Stage 4: Generating {len(routes.routes)-1} direct transitions (region-only mode)")
            transitions: List[List[Waypoint]] = []
            for i in range(len(routes.routes) - 1):
                start_wp = routes.routes[i][-1]
                goal_wp = routes.routes[i + 1][0]
                transition = self._build_region_only_direct_transition(start_wp, goal_wp)
                transitions.append(transition)
                logger.info(f"  Transition {i+1}: {len(transition)} waypoints (direct)")
            return transitions

        logger.info(f"Stage 4: Generating {len(routes.routes)-1} transitions with Theta*")
        
        generator = TransitionThetaStarGenerator(
            grid_resolution=self.config.transition_grid_resolution,
            max_expansions=self.config.transition_max_expansions,
            safety_margin=self.config.safety_distance,
            goal_tolerance=self.config.transition_goal_tolerance,
            waypoint_spacing=self.config.transition_waypoint_spacing,
        )
        
        transitions = []
        for i in range(len(routes.routes) - 1):
            start_wp = routes.routes[i][-1]  # List[Waypoint], access last element
            goal_wp = routes.routes[i+1][0]  # List[Waypoint], access first element
            
            config = GeneratorConfig(
                start_waypoint=start_wp,
                goal_waypoint=goal_wp,
                transition_grid_resolution=self.config.transition_grid_resolution,
                transition_max_expansions=self.config.transition_max_expansions,
                transition_goal_tolerance=self.config.transition_goal_tolerance,
                transition_waypoint_spacing=self.config.transition_waypoint_spacing,
                transition_prefer_lateral_before_altitude=self.config.transition_prefer_lateral_before_altitude,
                transition_lateral_offset_min_m=self.config.transition_lateral_offset_min_m,
                transition_lateral_offset_max_m=self.config.transition_lateral_offset_max_m,
                transition_lateral_offset_step_m=self.config.transition_lateral_offset_step_m,
                transition_lateral_max_candidates=self.config.transition_lateral_max_candidates,
                transition_lateral_turn_penalty_weight=self.config.transition_lateral_turn_penalty_weight,
                transition_enable_theta_star_fallback=self.config.transition_enable_theta_star_fallback,
            )
            
            # Pass cached obstacle tree to avoid redundant cKDTree construction
            transition = generator.generate(self.pointcloud, self.config.camera, config, self._obstacle_pointcloud_tree)[0]
            transitions.append(transition)
            logger.info(f"  Transition {i+1}: {len(transition)} waypoints")
        
        return transitions

    def _build_region_only_direct_transition(self, start_wp: Waypoint, goal_wp: Waypoint) -> List[Waypoint]:
        """Build a simple direct transition for region-only mode."""
        start_pos = np.array([start_wp.x, start_wp.y, start_wp.z], dtype=float)
        goal_pos = np.array([goal_wp.x, goal_wp.y, goal_wp.z], dtype=float)
        delta = goal_pos - start_pos
        distance = float(np.linalg.norm(delta))

        if distance < 1e-6:
            heading = float(goal_wp.heading_deg)
        else:
            heading = float(np.degrees(np.arctan2(delta[0], delta[1])))

        midpoint = Waypoint(
            x=float((start_wp.x + goal_wp.x) * 0.5),
            y=float((start_wp.y + goal_wp.y) * 0.5),
            z=float((start_wp.z + goal_wp.z) * 0.5),
            heading_deg=heading,
            gimbal_pitch_deg=float((start_wp.gimbal_pitch_deg + goal_wp.gimbal_pitch_deg) * 0.5),
            speed_ms=float((start_wp.speed_ms + goal_wp.speed_ms) * 0.5),
            action=WaypointAction.HOVER,
            dwell_time_s=0.0,
            is_keypoint=False,
            waypoint_type="transition",
        )

        goal_transition = Waypoint(
            x=float(goal_wp.x),
            y=float(goal_wp.y),
            z=float(goal_wp.z),
            heading_deg=heading,
            gimbal_pitch_deg=float(goal_wp.gimbal_pitch_deg),
            speed_ms=float(goal_wp.speed_ms),
            action=WaypointAction.HOVER,
            dwell_time_s=0.0,
            is_keypoint=True,
            waypoint_type="transition",
        )

        return [start_wp, midpoint, goal_transition]

    def _postprocess_trajectory_for_capture(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Interpolate by segment and generate capture points only on coverage segments."""
        if len(waypoints) <= 1:
            return waypoints

        def _point_key(wp: Waypoint) -> tuple:
            return (
                round(float(wp.x), 3),
                round(float(wp.y), 3),
                round(float(wp.z), 3),
                getattr(wp, "parent_route_id", None),
            )

        chunks: List[List[Waypoint]] = []
        current_chunk: List[Waypoint] = [waypoints[0]]
        current_type = getattr(waypoints[0], "waypoint_type", "coverage")

        for waypoint in waypoints[1:]:
            waypoint_type = getattr(waypoint, "waypoint_type", "coverage")
            if waypoint_type == current_type:
                current_chunk.append(waypoint)
                continue

            chunks.append(current_chunk)
            current_chunk = [waypoint]
            current_type = waypoint_type

        chunks.append(current_chunk)

        processed: List[Waypoint] = []
        for chunk in chunks:
            chunk_type = getattr(chunk[0], "waypoint_type", "coverage")
            raw_keypoints = {_point_key(wp) for wp in chunk if wp.is_keypoint}
            default_parent_route = getattr(chunk[0], "parent_route_id", None)

            if chunk_type == "coverage":
                interpolated_chunk = self.capture_sampler.interpolate_waypoints(
                    chunk,
                    self.pointcloud,
                    self.config.camera,
                    self.config.altitude,
                    self.config.front_overlap,
                    self.config.speed_ms,
                    interpolation_factor=self.config.capture_interpolation_factor,
                )
            else:
                interpolated_chunk = [wp for wp in chunk]

            for waypoint in interpolated_chunk:
                waypoint.waypoint_type = chunk_type
                if waypoint.parent_route_id is None:
                    waypoint.parent_route_id = default_parent_route

                if chunk_type == "transition":
                    waypoint.action = WaypointAction.HOVER

                waypoint.is_keypoint = _point_key(waypoint) in raw_keypoints

            if chunk_type == "coverage" and interpolated_chunk:
                capture_indices = self.capture_sampler.mark_capture_points(
                    interpolated_chunk,
                    self.pointcloud,
                    self.config.camera,
                    self.config.altitude,
                    self.config.front_overlap,
                    self.config.speed_ms,
                )
                for index in capture_indices:
                    interpolated_chunk[index].action = WaypointAction.SHOOT

                for waypoint in interpolated_chunk:
                    if waypoint.is_keypoint:
                        waypoint.action = WaypointAction.SHOOT

                interpolated_chunk = [
                    waypoint for waypoint in interpolated_chunk
                    if waypoint.action == WaypointAction.SHOOT
                ]

            if not processed:
                processed.extend(interpolated_chunk)
            elif interpolated_chunk:
                processed.extend(interpolated_chunk[1:])

        return processed

class CoveragePlanner:
    """High-level planner interface.
    
    Convenience wrapper around CoveragePipeline for simpler API.
    Uses DataLoader for unified data loading.
    
    Example:
        planner = CoveragePlanner(config)
        result = planner.plan()
    """
    
    def __init__(self, config: MissionConfig):
        """Initialize planner.
        
        Args:
            config: Mission configuration
        """
        self.config = config
        self._data_loader = DataLoader(config)
    
    def plan(self) -> PlannerResult:
        """Execute planning.
        
        Loads point cloud and runs pipeline.
        
        Returns:
            Planning result
        """
        # Load target/obstacle point clouds using DataLoader
        target_pc, obstacle_pc = self._data_loader.load()
        
        # Run pipeline
        pipeline = CoveragePipeline(self.config)
        return pipeline.run(target_pc, obstacle_pc)
