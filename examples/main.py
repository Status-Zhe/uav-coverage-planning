from uav_planners import CoveragePlanner, MissionConfig
from uav_planners.models import Camera
import numpy as np
import time

start = time.time()
print(f"[{time.time()-start:.1f}s] Starting...")

# Unified input path (auto-detect)
# - .json => tileset mode
# - .ply/.pcd => pointcloud_file mode
INPUT_DATA_PATH = "/home/xz/cr_nanobot/.nanobot/workspace/uav-planning/pointcloud/zijinCompany-2025-11-27-all/3d_tiles/pointcloud/tileset.json"
# INPUT_DATA_PATH = "/home/xz/cr_nanobot/.nanobot/workspace/uav-planning/pointcloud/reconstruction_clean.ply"
POINTCLOUD_PREPROCESS_ENABLE = True  # 预处理点云
GLOBAL_DISTANCE_M = 5
MIN_FLIGHT_ALTITUDE_M = 2.0
TARGET_GSD_M_PER_PIXEL = 0.01  # e.g. 0.01 means 1cm/pixel; when set, overrides GLOBAL_DISTANCE_M
WAYPOINT_EXPORT_FRAME = "planning"  # planning | source

# 输入AABB box中心和 长宽高 切块点云，只对方块中的点云做规划（在内置 PointCloud 对象层执行）
# 方式A（直接传矩阵，推荐矩阵用 numpy）：
# AABB_TRANSFORM = None
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

# Configure mission
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
    algorithm="viewpoint_optimized", # boustrophedon or oblique or spiral or viewpoint_optimized or viewpoint_wrap
    altitude=23.0,
    speed_ms=5.0,
    side_overlap=0.5,
    safety_distance=3.0,
    voxel_size=1.0,  # Voxel grid resolution for collision detection (meters)
    capture_interpolation_factor=1.0  # 0.1 ~ 3.0, smaller is denser
)

if AABB_SIZE is not None:
    if AABB_TRANSFORM is not None:
        config.aabb_transform = AABB_TRANSFORM
        config.aabb_size = AABB_SIZE
    elif AABB_CENTER is not None and AABB_YAW_DEG is not None:
        config.aabb_center = AABB_CENTER
        config.aabb_yaw_deg = AABB_YAW_DEG
        config.aabb_size = AABB_SIZE
    else:
        # transform/center+yaw 都未提供时，不进行切割
        config.aabb_transform = None
        config.aabb_center = None
        config.aabb_yaw_deg = 0.0
        config.aabb_size = None

# 3d-tiles support parameters
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

# transition parameters (Theta*)
config.transition_grid_resolution=1.0
config.transition_max_expansions=30000
config.transition_goal_tolerance=2.0
config.transition_waypoint_spacing=4.0
config.transition_enable_theta_star_fallback=True

# oblique
config.oblique_dst_srf = config.global_distance_m  # Set oblique-specific parameter
config.oblique_min_altitude = config.min_flight_altitude_m  # backward compatibility; unified min altitude uses min_flight_altitude_m

# viewpoint_optimized
config.viewpoint_boundary_expand_m=config.global_distance_m    # 每层边界外扩距离
config.viewpoint_min_points_per_layer=4   # 每层至少4个点
config.viewpoint_layer_order="bottom_up"  # or "top_down" 从上到下 or 从下到上
config.viewpoint_min_altitude = config.min_flight_altitude_m       # backward compatibility; unified min altitude uses min_flight_altitude_m
config.viewpoint_beyond_altitude = 3.0    # 最高点以上再飞3米，增加额外覆盖（可设0不增加）
config.viewpoint_layer_height_step_m=3.3  # 层高
config.viewpoint_ring_arc_step_m=10      # 圆弧步长
config.viewpoint_shape_use_full_points=True  # False=速度优先, True=边界精度优先（alpha输入点集）
config.viewpoint_shape_method="convex"  # alpha 或 convex
config.viewpoint_shape_roundness_m=0.8 # 轮廓圆滑参数（对 alpha 结果同样生效）
config.viewpoint_alpha=2.0  # alpha shape参数: 越小越贴凹形，越大越接近外轮廓shape

# viewpoint_optimized inserted layer control 
config.viewpoint_layer_area_jump_ratio=3.0  # 面积跳变触发阈值（>1.0）
config.viewpoint_layer_insert_max_global=8  # 全局最多插入的补间层数

# Run planning
print(f"[{time.time()-start:.1f}s] Creating planner...")
pipeline = CoveragePlanner(config)

print(f"[{time.time()-start:.1f}s] Running plan()...")
result = pipeline.plan()

# Results
print(f"[{time.time()-start:.1f}s] Plan complete!")
print(f"[{time.time()-start:.1f}s] Generated {result.keypoint_count} keypoints")
print(f"[{time.time()-start:.1f}s] Total distance: {result.to_trajectory().total_distance_m:.1f} m")

# Export
print(f"[{time.time()-start:.1f}s] Exporting...")
result.export_json("trajectory2.json")
result.export_csv("waypoints2.csv")
print(f"[{time.time()-start:.1f}s] Done!")