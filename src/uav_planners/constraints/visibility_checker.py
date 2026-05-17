"""Visibility checking for viewpoint planning."""

import math
from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree

from ..models.waypoint import Waypoint
from ..models.pointcloud import PointCloud


class VisibilityChecker:
    """Check visibility between viewpoints and target points.
    
    Uses ray casting to determine which points are visible from a given
    viewpoint, considering occlusions from the structure itself.
    
    Attributes:
        pointcloud: Target point cloud
        kdtree: KD-tree for efficient nearest neighbor queries
        min_range: Minimum viewing distance
        max_range: Maximum viewing distance
    """
    
    def __init__(
        self,
        pointcloud: PointCloud,
        min_range: float = 2.0,
        max_range: float = 200.0
    ):
        """Initialize visibility checker.
        
        Args:
            pointcloud: Target point cloud (obstacles + targets)
            min_range: Minimum valid viewing distance
            max_range: Maximum valid viewing distance
        """
        self.pointcloud = pointcloud
        self.min_range = min_range
        self.max_range = max_range
        self.kdtree = cKDTree(pointcloud.points)
    
    def check_visibility(
        self,
        viewpoint: Waypoint,
        target_points: Optional[PointCloud] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Check which target points are visible from viewpoint.
        
        Args:
            viewpoint: Camera position and orientation
            target_points: Points to check (default: all points in pointcloud)
            
        Returns:
            Tuple of (visible_indices, distances)
        """
        if target_points is None:
            target_points = self.pointcloud
        
        vp_pos = np.array([viewpoint.x, viewpoint.y, viewpoint.z])
        
        # Get camera direction vector
        heading_rad = math.radians(viewpoint.heading_deg)
        pitch_rad = math.radians(viewpoint.gimbal_pitch_deg)
        
        cam_dir = np.array([
            math.sin(heading_rad) * math.cos(-pitch_rad),
            math.cos(heading_rad) * math.cos(-pitch_rad),
            math.sin(-pitch_rad)
        ])
        
        # Check all target points
        visible_indices = []
        distances = []
        
        for i, target in enumerate(target_points.points):
            is_visible, distance = self._check_single_visibility(
                vp_pos, cam_dir, target
            )
            
            if is_visible:
                visible_indices.append(i)
                distances.append(distance)
        
        return np.array(visible_indices), np.array(distances)
    
    def _check_single_visibility(
        self,
        vp_pos: np.ndarray,
        cam_dir: np.ndarray,
        target: np.ndarray
    ) -> Tuple[bool, float]:
        """Check visibility of a single target point.
        
        Args:
            vp_pos: Viewpoint position
            cam_dir: Camera direction (normalized)
            target: Target point position
            
        Returns:
            Tuple of (is_visible, distance)
        """
        # Vector to target
        to_target = target - vp_pos
        distance = np.linalg.norm(to_target)
        
        # Check distance range
        if distance < self.min_range or distance > self.max_range:
            return False, distance
        
        # Normalize direction to target
        to_target_norm = to_target / distance
        
        # Check if within camera FOV (simplified: cone check)
        # Cosine of angle between camera direction and target direction
        cos_angle = np.dot(cam_dir, to_target_norm)
        
        # If angle > 60 degrees (cos < 0.5), not visible
        if cos_angle < 0.5:  # cos(60°) = 0.5
            return False, distance
        
        # Ray casting for occlusion
        if self._is_occluded(vp_pos, target):
            return False, distance
        
        return True, distance
    
    def _is_occluded(self, vp_pos: np.ndarray, target: np.ndarray) -> bool:
        """Check if target is occluded by any point in point cloud.
        
        Uses ray-sphere intersection test with adaptive radius.
        
        Args:
            vp_pos: Viewpoint position
            target: Target point position
            
        Returns:
            True if occluded
        """
        # Direction and distance
        direction = target - vp_pos
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return False
        
        direction = direction / distance
        
        # Sample points along ray
        num_samples = max(10, int(distance / 2.0))  # Sample every 2m
        t_values = np.linspace(0.1, 0.9, num_samples)  # Avoid endpoints
        
        # Occlusion radius (adaptive to point density)
        occlusion_radius = max(0.5, self._estimate_point_spacing())
        
        for t in t_values:
            point_on_ray = vp_pos + t * distance * direction
            
            # Check if any point is near this ray point
            if self._has_nearby_point(point_on_ray, occlusion_radius):
                return True
        
        return False
    
    def _has_nearby_point(self, position: np.ndarray, radius: float) -> bool:
        """Check if there's any point within radius of position.
        
        Args:
            position: Query position
            radius: Search radius
            
        Returns:
            True if point found
        """
        distance, _ = self.kdtree.query(position, k=1)
        return distance < radius
    
    def _estimate_point_spacing(self) -> float:
        """Estimate average spacing between points.
        
        Returns:
            Estimated spacing in meters
        """
        # Sample some points and find nearest neighbor distances
        n_samples = min(100, self.pointcloud.point_count)
        indices = np.random.choice(
            self.pointcloud.point_count,
            size=n_samples,
            replace=False
        )
        
        sample_points = self.pointcloud.points[indices]
        distances, _ = self.kdtree.query(sample_points, k=2)
        
        # k=2 returns [self-distance, nearest-neighbor-distance]
        # Use the second column (index 1)
        return float(np.median(distances[:, 1]))
    
    def compute_view_score(
        self,
        viewpoint: Waypoint,
        target_points: PointCloud
    ) -> float:
        """Compute a quality score for a viewpoint.
        
        Higher score = better viewpoint.
        
        Args:
            viewpoint: Viewpoint to evaluate
            target_points: Points to cover
            
        Returns:
            Quality score (0-1)
        """
        visible_indices, distances = self.check_visibility(viewpoint, target_points)
        
        if len(visible_indices) == 0:
            return 0.0
        
        # Factors:
        # 1. Number of visible points
        coverage_ratio = len(visible_indices) / target_points.point_count
        
        # 2. Average distance (closer is better)
        avg_distance = np.mean(distances)
        distance_score = 1.0 / (1.0 + avg_distance / 50.0)  # Normalize
        
        # Combined score
        score = 0.7 * coverage_ratio + 0.3 * distance_score
        
        return float(score)
