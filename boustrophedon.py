from src.uav_planners import CoveragePlanner, MissionConfig
from src.uav_planners.models import Camera
import time

start = time.time()
print(f"[{time.time()-start:.1f}s] Starting region-only planner...")

# Region-only boustrophedon settings (no pointcloud / no tileset)
REGION_RECT_XY = (-120.0, -80.0, 120.0, 80.0)  # xmin, ymin, xmax, ymax
REGION_GROUND_Z = 0.0
ALTITUDE_M = 40.0
GLOBAL_DISTANCE_M = 12.0
MIN_FLIGHT_ALTITUDE_M = 5.0
WAYPOINT_EXPORT_FRAME = "planning"  # planning | source

config = MissionConfig(
    pointcloud_path=None,
    data_source_type="auto",
    region_only_enabled=True,
    coverage_area_rect_xy=REGION_RECT_XY,
    region_ground_z=REGION_GROUND_Z,
    camera=Camera(
        focal_length_mm=24.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        resolution_x=6000,
        resolution_y=4000,
    ),
    algorithm="boustrophedon",
    altitude=ALTITUDE_M,
    speed_ms=5.0,
    side_overlap=0.5,
    front_overlap=0.8,
    global_distance_m=GLOBAL_DISTANCE_M,
    min_flight_altitude_m=MIN_FLIGHT_ALTITUDE_M,
    capture_interpolation_factor=1.0,
)

# In region-only mode transitions are direct-line joins and collision stages are skipped.
config.transition_waypoint_spacing = 4.0
config.waypoint_export_frame = WAYPOINT_EXPORT_FRAME

print(f"[{time.time()-start:.1f}s] Creating planner...")
planner = CoveragePlanner(config)

print(f"[{time.time()-start:.1f}s] Running plan()...")
result = planner.plan()

print(f"[{time.time()-start:.1f}s] Plan complete!")
print(f"[{time.time()-start:.1f}s] Generated {result.keypoint_count} keypoints")
print(f"[{time.time()-start:.1f}s] Total distance: {result.to_trajectory().total_distance_m:.1f} m")

print(f"[{time.time()-start:.1f}s] Exporting...")
result.export_json("trajectory_boustrophedon.json")
result.export_csv("waypoints_boustrophedon.csv")
print(f"[{time.time()-start:.1f}s] Done!")
