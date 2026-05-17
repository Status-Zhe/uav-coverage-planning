from src.uav_planners import CoveragePlanner, MissionConfig
from src.uav_planners.models import Camera
import time

start = time.time()
print(f"[{time.time()-start:.1f}s] Starting...")

GLOBAL_DISTANCE_M = 8.0
SIDE_OVERLAP = 0.6

polygon_xyz = [
    (0.0, 0.0, 2.0),
    (20.0, 0.0, 6.0),
    (20.0, 10.0, 7.0),
    (0.0, 10.0, 3.0),
]

config = MissionConfig(
    pointcloud_path=None,
    data_source_type="auto",
    camera=Camera(
        focal_length_mm=24.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        resolution_x=6000,
        resolution_y=4000,
    ),
    algorithm="oblique_oneplane",
    altitude=20.0,
    speed_ms=5.0,
    side_overlap=SIDE_OVERLAP,
    safety_distance=3.0,
    global_distance_m=GLOBAL_DISTANCE_M,
    oneplane_polygon_xyz=polygon_xyz,
    oneplane_plane_tolerance=0.02,
    oneplane_face_normal_sign=-1.0,
    oneplane_heading_yaw_offset_deg=0.0,
)

print(f"[{time.time()-start:.1f}s] Creating planner...")
pipeline = CoveragePlanner(config)

print(f"[{time.time()-start:.1f}s] Running plan()...")
result = pipeline.plan()

print(f"[{time.time()-start:.1f}s] Plan complete!")
print(f"[{time.time()-start:.1f}s] Generated {result.keypoint_count} keypoints")
print(f"[{time.time()-start:.1f}s] Total distance: {result.to_trajectory().total_distance_m:.1f} m")

print(f"[{time.time()-start:.1f}s] Exporting...")
result.export_json("trajectory_oblique_oneplane.json")
result.export_csv("waypoints_oblique_oneplane.csv")
print(f"[{time.time()-start:.1f}s] Done!")
