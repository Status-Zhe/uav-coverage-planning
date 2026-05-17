from src.uav_planners import CoveragePlanner, MissionConfig
from src.uav_planners.models import Camera
import time

start = time.time()
print(f"[{time.time()-start:.1f}s] Starting...")

GLOBAL_DISTANCE_M = 12
MIN_FLIGHT_ALTITUDE_M = 2.0
TARGET_GSD_M_PER_PIXEL = 0.01

config = MissionConfig(
    pointcloud_path=None,
    gsd=TARGET_GSD_M_PER_PIXEL,
    global_distance_m=GLOBAL_DISTANCE_M,
    min_flight_altitude_m=MIN_FLIGHT_ALTITUDE_M,
    data_source_type="auto",
    camera=Camera(
        focal_length_mm=24.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        resolution_x=6000,
        resolution_y=4000,
    ),
    algorithm="spiral",
    altitude=23.0,
    speed_ms=5.0,
    side_overlap=0.5,
    safety_distance=3.0,
    voxel_size=1.0,
    capture_interpolation_factor=1.0,
    spiral_center_xy=(0.0, 0.0),
    spiral_radius=12.0,
    spiral_start_z=2.0,
    spiral_height=30.0,
)

config.transition_grid_resolution = 1.0
config.transition_max_expansions = 30000
config.transition_goal_tolerance = 2.0
config.transition_waypoint_spacing = 4.0
config.transition_enable_theta_star_fallback = True

print(f"[{time.time()-start:.1f}s] Creating planner...")
pipeline = CoveragePlanner(config)

print(f"[{time.time()-start:.1f}s] Running plan()...")
result = pipeline.plan()

print(f"[{time.time()-start:.1f}s] Plan complete!")
print(f"[{time.time()-start:.1f}s] Generated {result.keypoint_count} keypoints")
print(f"[{time.time()-start:.1f}s] Total distance: {result.to_trajectory().total_distance_m:.1f} m")

print(f"[{time.time()-start:.1f}s] Exporting...")
result.export_json("trajectory_spiral_nopc.json")
result.export_csv("waypoints_spiral_nopc.csv")
print(f"[{time.time()-start:.1f}s] Done!")
