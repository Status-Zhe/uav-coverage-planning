from src.uav_planners import CoveragePlanner, MissionConfig
from src.uav_planners.models import Camera
import numpy as np
import time

start = time.time()
print(f"[{time.time()-start:.1f}s] Starting...")

INPUT_DATA_PATH = "/home/xz/cr_nanobot/.nanobot/workspace/uav-planning/pointcloud/reconstruction_clean.ply"
POINTCLOUD_PREPROCESS_ENABLE = True
GLOBAL_DISTANCE_M = 5
MIN_FLIGHT_ALTITUDE_M = 2.0
TARGET_GSD_M_PER_PIXEL = 0.01
WAYPOINT_EXPORT_FRAME = "planning"

config = MissionConfig(
    pointcloud_path=INPUT_DATA_PATH,
    pointcloud_preprocess_enable=POINTCLOUD_PREPROCESS_ENABLE,
    gsd=TARGET_GSD_M_PER_PIXEL,
    global_distance_m=GLOBAL_DISTANCE_M,
    min_flight_altitude_m=MIN_FLIGHT_ALTITUDE_M,
    data_source_type="auto",
    camera=Camera(
        focal_length_mm=24.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        resolution_x=6000,
        resolution_y=4000
    ),
    algorithm="viewpoint_optimized",
    altitude=23.0,
    speed_ms=5.0,
    side_overlap=0.5,
    safety_distance=3.0,
    voxel_size=1.0,
    capture_interpolation_factor=1.0
)

config.transition_grid_resolution=1.0
config.transition_max_expansions=30000
config.transition_goal_tolerance=2.0
config.transition_waypoint_spacing=4.0
config.transition_enable_theta_star_fallback=True

config.viewpoint_boundary_expand_m=config.global_distance_m
config.viewpoint_min_points_per_layer=4
config.viewpoint_layer_order="bottom_up"
config.viewpoint_min_altitude = config.min_flight_altitude_m
config.viewpoint_beyond_altitude = 3.0
config.viewpoint_layer_height_step_m=3.3
config.viewpoint_ring_arc_step_m=10
config.viewpoint_shape_use_full_points=True
config.viewpoint_shape_method="convex"  # convex | alpha
config.viewpoint_shape_roundness_m=0.8
config.viewpoint_alpha=2.0
config.viewpoint_layer_area_jump_ratio=3.0
config.viewpoint_layer_insert_max_global=8

print(f"[{time.time()-start:.1f}s] Creating planner...")
pipeline = CoveragePlanner(config)

print(f"[{time.time()-start:.1f}s] Running plan()...")
result = pipeline.plan()

print(f"[{time.time()-start:.1f}s] Plan complete!")
print(f"[{time.time()-start:.1f}s] Generated {result.keypoint_count} keypoints")
print(f"[{time.time()-start:.1f}s] Total distance: {result.to_trajectory().total_distance_m:.1f} m")

print(f"[{time.time()-start:.1f}s] Exporting...")
result.export_json("trajectory_viewpoint_optimized.json")
result.export_csv("waypoints_viewpoint_optimized.csv")
print(f"[{time.time()-start:.1f}s] Done!")
