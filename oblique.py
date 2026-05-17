from src.uav_planners import CoveragePlanner, MissionConfig
from src.uav_planners.models import Camera
import numpy as np
import time

start = time.time()
print(f"[{time.time()-start:.1f}s] Starting...")

INPUT_DATA_PATH = "/home/xz/cr_nanobot/.nanobot/workspace/uav-planning/pointcloud/zijinCompany-2025-11-27-all/3d_tiles/pointcloud/tileset.json"
POINTCLOUD_PREPROCESS_ENABLE = True  # 预处理点云
GLOBAL_DISTANCE_M = 5
MIN_FLIGHT_ALTITUDE_M = 2.0
TARGET_GSD_M_PER_PIXEL = 0.01
WAYPOINT_EXPORT_FRAME = "planning" # planning | source

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
    algorithm="oblique",
    altitude=23.0,
    speed_ms=5.0,
    side_overlap=0.5,
    safety_distance=3.0,
    voxel_size=1.0,
    capture_interpolation_factor=1.0
)

# 如果需要对规划点云进行切割，配置AABB参数（方式A或方式B），否则保持默认None禁用切割。
# 方式A（直接传矩阵，推荐矩阵用 numpy）：
AABB_TRANSFORM = np.array([
    [-0.444108784199, -0.895970165730, 0.0, -98.991455078125],
    [0.895970165730, -0.444108784199, 0.0, -30.301025390625],
    [0.0, 0.0, 0.999995172024, 35],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)
AABB_SIZE = (43.0, 70.0, 60)     # (sx, sy, sz); set None to disable AABB crop

# 方式B（传中心+偏航角，内部转换矩阵）：
AABB_CENTER = None
AABB_YAW_DEG = None
# AABB_CENTER = (-98.991455078125, -30.301025390625, 30.939636230469)
# AABB_YAW_DEG = 116.36433307288706  # XY平面偏航角(度)
# AABB_SIZE = (43.0, 70.0, 65.7)     # (sx, sy, sz); set None to disable AABB crop

if AABB_SIZE is not None:
    if AABB_TRANSFORM is not None:
        config.aabb_transform = AABB_TRANSFORM
        config.aabb_size = AABB_SIZE
    elif AABB_CENTER is not None and AABB_YAW_DEG is not None:
        config.aabb_center = AABB_CENTER
        config.aabb_yaw_deg = AABB_YAW_DEG
        config.aabb_size = AABB_SIZE
    else:
        config.aabb_transform = None
        config.aabb_center = None
        config.aabb_yaw_deg = 0.0
        config.aabb_size = None

# 如果格式是3d-tiles,配置以下参数
config.tiles_kind="auto"
config.tiles_max_points=5000000
config.tiles_lod_max=3
config.tiles_output_frame="enu"
config.tiles_input_crs="ecef"
config.waypoint_export_frame = WAYPOINT_EXPORT_FRAME
# Optional: convert tileset to PLY first, then load PLY for planning
# config.tiles_convert_to_ply=True,
# config.tiles_converted_ply_path="/home/xz/cr_nanobot/.nanobot/workspace/uav-planning/pointcloud/tileset_converted_enu.ply",
# config.tiles_convert_coord_frame="enu",
# config.tiles_convert_write_ascii=True,


# Transition settings (for non-region-only modes)
config.transition_grid_resolution=1.0
config.transition_max_expansions=30000
config.transition_goal_tolerance=2.0
config.transition_waypoint_spacing=4.0
config.transition_enable_theta_star_fallback=True

# Oblique-specific settings
config.oblique_dst_srf = config.global_distance_m
config.oblique_min_altitude = config.min_flight_altitude_m

print(f"[{time.time()-start:.1f}s] Creating planner...")
pipeline = CoveragePlanner(config)

print(f"[{time.time()-start:.1f}s] Running plan()...")
result = pipeline.plan()

print(f"[{time.time()-start:.1f}s] Plan complete!")
print(f"[{time.time()-start:.1f}s] Generated {result.keypoint_count} keypoints")
print(f"[{time.time()-start:.1f}s] Total distance: {result.to_trajectory().total_distance_m:.1f} m")

print(f"[{time.time()-start:.1f}s] Exporting...")
result.export_json("trajectory_oblique.json")
result.export_csv("waypoints_oblique.csv")
print(f"[{time.time()-start:.1f}s] Done!")
