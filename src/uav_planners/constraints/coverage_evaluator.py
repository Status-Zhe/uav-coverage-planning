"""Coverage evaluation for generated trajectories."""

import math
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.spatial import cKDTree

from ..models.waypoint import Waypoint
from ..models.pointcloud import PointCloud
from ..models.camera import Camera


class CoverageEvaluator:
    """Evaluate coverage quality of a trajectory.
    
    Computes how well the waypoints cover the target point cloud,
    considering camera field of view and visibility.
    
    Attributes:
        coverage_threshold: Minimum required coverage ratio
    """
    
    def __init__(self, coverage_threshold: float = 0.95):
        """Initialize coverage evaluator.
        
        Args:
            coverage_threshold: Required coverage ratio (0-1)
        """
        self.coverage_threshold = coverage_threshold
    
    def evaluate(
        self,
        waypoints: List[Waypoint],
        pointcloud: PointCloud,
        camera: Camera
    ) -> Dict[str, Any]:
        """Evaluate coverage of waypoints over point cloud.
        
        Args:
            waypoints: Planned waypoints
            pointcloud: Target point cloud to cover
            camera: Camera specification
            
        Returns:
            Coverage report dictionary with:
            - coverage_ratio: Fraction of points covered
            - total_points: Total number of target points
            - covered_points: Number of covered points
            - uncovered_indices: Indices of uncovered points
            - average_views: Average views per point
        """
        if not waypoints or pointcloud.point_count == 0:
            return {
                "coverage_ratio": 0.0,
                "total_points": pointcloud.point_count,
                "covered_points": 0,
                "uncovered_indices": list(range(pointcloud.point_count)),
                "average_views": 0.0,
            }
        
        # Compute coverage for each point
        coverage_counts = self._compute_point_coverage(
            waypoints, pointcloud, camera
        )
        
        # Calculate statistics
        covered_mask = coverage_counts > 0
        covered_points = int(np.sum(covered_mask))
        total_points = pointcloud.point_count
        coverage_ratio = covered_points / total_points if total_points > 0 else 0.0
        
        uncovered_indices = np.where(~covered_mask)[0].tolist()
        average_views = float(np.mean(coverage_counts)) if total_points > 0 else 0.0
        
        return {
            "coverage_ratio": coverage_ratio,
            "total_points": total_points,
            "covered_points": covered_points,
            "uncovered_indices": uncovered_indices,
            "average_views": average_views,
            "coverage_threshold": self.coverage_threshold,
            "meets_threshold": coverage_ratio >= self.coverage_threshold,
        }
    
    def _compute_point_coverage(
        self,
        waypoints: List[Waypoint],
        pointcloud: PointCloud,
        camera: Camera
    ) -> np.ndarray:
        """Compute how many waypoints cover each point.
        
        Args:
            waypoints: Planned viewpoints
            pointcloud: Target points
            camera: Camera specification
            
        Returns:
            Array of coverage counts per point
        """
        coverage_counts = np.zeros(pointcloud.point_count, dtype=int)
        
        # Pre-compute camera parameters
        fov_h_rad = math.radians(camera.fov_horizontal_deg)
        fov_v_rad = math.radians(camera.fov_vertical_deg)
        max_distance = self._estimate_max_range(camera)
        
        for wp in waypoints:
            visible_indices = self._get_visible_points(
                wp, pointcloud, fov_h_rad, fov_v_rad, max_distance
            )
            coverage_counts[visible_indices] += 1
        
        return coverage_counts
    
    def _get_visible_points(
        self,
        waypoint: Waypoint,
        pointcloud: PointCloud,
        fov_h_rad: float,
        fov_v_rad: float,
        max_distance: float
    ) -> np.ndarray:
        """Get indices of points visible from a waypoint.
        
        Args:
            waypoint: Viewpoint
            pointcloud: Target points
            fov_h_rad: Horizontal FOV in radians
            fov_v_rad: Vertical FOV in radians
            max_distance: Maximum viewing distance
            
        Returns:
            Indices of visible points
        """
        wp_pos = np.array([waypoint.x, waypoint.y, waypoint.z])
        
        # Vector to all points
        vectors = pointcloud.points - wp_pos
        distances = np.linalg.norm(vectors, axis=1)
        
        # Filter by distance
        within_range = distances <= max_distance
        
        if not np.any(within_range):
            return np.array([], dtype=int)
        
        # Compute camera direction vector
        heading_rad = math.radians(waypoint.heading_deg)
        pitch_rad = math.radians(waypoint.gimbal_pitch_deg)
        
        # Camera forward direction
        cam_dir = np.array([
            math.sin(heading_rad) * math.cos(-pitch_rad),
            math.cos(heading_rad) * math.cos(-pitch_rad),
            math.sin(-pitch_rad)
        ])
        
        # Normalize vectors to points
        valid_indices = np.where(within_range)[0]
        valid_vectors = vectors[valid_indices]
        valid_distances = distances[valid_indices].reshape(-1, 1)
        
        # Avoid division by zero
        valid_distances = np.maximum(valid_distances, 1e-6)
        normalized = valid_vectors / valid_distances
        
        # Compute angles
        cos_angles = np.dot(normalized, cam_dir)
        angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))
        
        # Check if within FOV (approximate as cone)
        fov_cone = max(fov_h_rad, fov_v_rad) / 2
        visible = angles <= fov_cone
        
        return valid_indices[visible]
    
    def _estimate_max_range(self, camera: Camera) -> float:
        """Estimate maximum useful viewing distance.
        
        Based on GSD requirements - typically want GSD < 5cm for inspection.
        
        Args:
            camera: Camera specification
            
        Returns:
            Maximum viewing distance in meters
        """
        # Conservative estimate: max altitude where GSD < 5cm
        # This is a simplified calculation
        return 200.0  # 200m default max range
    
    def identify_uncovered_regions(
        self,
        waypoints: List[Waypoint],
        pointcloud: PointCloud,
        camera: Camera
    ) -> List[Tuple[float, float, float]]:
        """Identify 3D regions that are not well covered.
        
        Args:
            waypoints: Planned waypoints
            pointcloud: Target point cloud
            camera: Camera specification
            
        Returns:
            List of uncovered region centers
        """
        report = self.evaluate(waypoints, pointcloud, camera)
        
        if not report["uncovered_indices"]:
            return []
        
        # Get uncovered points
        uncovered_points = pointcloud.points[report["uncovered_indices"]]
        
        # Cluster uncovered points to find regions
        # Simple approach: return centroid of uncovered points
        if len(uncovered_points) > 0:
            centroid = np.mean(uncovered_points, axis=0)
            return [tuple(centroid)]
        
        return []
