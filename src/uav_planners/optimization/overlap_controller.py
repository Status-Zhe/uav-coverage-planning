"""Overlap consistency controller for uniform coverage scanning.

This module ensures consistent image overlap and capture distance
across complex building structures.
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional, List

import numpy as np

logger = logging.getLogger(__name__)


class OverlapController:
    """Controls overlap consistency for uniform coverage.
    
    Ensures that:
    - Front overlap (along path) is consistent
    - Side overlap (between adjacent paths) is consistent
    - Capture distance is uniform across the coverage area
    """
    
    def __init__(
        self,
        target_front_overlap: float = 0.3,
        target_side_overlap: float = 0.5,
        max_distance_variance: float = 0.2,
    ):
        """Initialize overlap controller.
        
        Args:
            target_front_overlap: Target front overlap ratio (0.0-1.0)
            target_side_overlap: Target side overlap ratio (0.0-1.0)
            max_distance_variance: Maximum allowed variance in capture distance
        """
        self.target_front_overlap = target_front_overlap
        self.target_side_overlap = target_side_overlap
        self.max_distance_variance = max_distance_variance
    
    def calculate_capture_distance(
        self,
        footprint_size: Tuple[float, float],
        camera_fov: Tuple[float, float],
        overlap: float,
    ) -> float:
        """Calculate consistent capture distance based on overlap.
        
        Args:
            footprint_size: (width, height) of camera footprint on surface
            camera_fov: (horizontal, vertical) field of view in degrees
            overlap: Overlap ratio (0.0-1.0)
        
        Returns:
            Recommended capture distance
        """
        footprint_width, footprint_height = footprint_size
        h_fov, v_fov = camera_fov
        
        # Calculate distance from footprint size and FOV
        # footprint_size = 2 * distance * tan(FOV/2)
        # distance = footprint_size / (2 * tan(FOV/2))
        
        h_distance = footprint_width / (2 * np.tan(np.radians(h_fov / 2)))
        v_distance = footprint_height / (2 * np.tan(np.radians(v_fov / 2)))
        
        # Use the larger distance to ensure both dimensions are covered
        base_distance = max(h_distance, v_distance)
        
        return base_distance
    
    def adjust_path_spacing(
        self,
        base_distance: float,
        actual_distance: float,
        overlap: float,
    ) -> float:
        """Adjust path spacing to maintain consistent overlap.
        
        Args:
            base_distance: Base capture distance
            actual_distance: Actual distance to surface
            overlap: Desired overlap ratio
        
        Returns:
            Adjusted path spacing
        """
        if actual_distance <= 0:
            return base_distance * (1 - overlap)
        
        # Scale path spacing based on distance ratio
        distance_ratio = actual_distance / base_distance
        spacing = base_distance * (1 - overlap) * distance_ratio
        
        return spacing
    
    def calculate_consistent_overlap(
        self,
        waypoints: List[np.ndarray],
        footprint_width: float,
    ) -> dict:
        """Calculate actual overlap from waypoint positions.
        
        Args:
            waypoints: List of waypoint positions
            footprint_width: Width of camera footprint
        
        Returns:
            Dictionary with overlap statistics
        """
        if len(waypoints) < 2:
            return {"mean_overlap": 1.0, "variance": 0.0, "min_overlap": 1.0}
        
        # Calculate distances between consecutive waypoints
        distances = []
        for i in range(len(waypoints) - 1):
            p1, p2 = waypoints[i], waypoints[i + 1]
            dist = np.linalg.norm(p1[:2] - p2[:2])
            distances.append(dist)
        
        mean_distance = np.mean(distances)
        
        # Calculate overlap from distance
        # overlap = 1 - (distance / footprint_width)
        overlaps = [1.0 - (d / footprint_width) for d in distances]
        
        return {
            "mean_overlap": np.mean(overlaps),
            "variance": np.var(overlaps),
            "min_overlap": np.min(overlaps),
            "max_overlap": np.max(overlaps),
            "mean_distance": mean_distance,
            "distance_variance": np.var(distances),
        }
    
    def validate_overlap_consistency(
        self,
        overlap_stats: dict,
    ) -> Tuple[bool, List[str]]:
        """Validate overlap consistency.
        
        Args:
            overlap_stats: Statistics from calculate_consistent_overlap
        
        Returns:
            Tuple of (is_valid, list of warnings)
        """
        warnings = []
        
        # Check variance
        if overlap_stats["variance"] > self.max_distance_variance:
            warnings.append(
                f"High overlap variance: {overlap_stats['variance']:.3f} "
                f"(threshold: {self.max_distance_variance})"
            )
        
        # Check minimum overlap
        if overlap_stats["min_overlap"] < 0.2:
            warnings.append(
                f"Low minimum overlap: {overlap_stats['min_overlap']:.3f} "
                "(may cause coverage gaps)"
            )
        
        is_valid = len(warnings) == 0
        return is_valid, warnings


class MultiLevelOverlapOptimizer:
    """Optimizes overlap for multi-level coverage planning.
    
    Handles different overlap requirements at different heights
    for complex building structures.
    """
    
    def __init__(
        self,
        base_overlap: float = 0.5,
        height_variance_compensation: float = 0.1,
    ):
        """Initialize multi-level optimizer.
        
        Args:
            base_overlap: Base overlap ratio
            height_variance_compensation: Compensation for height variance
        """
        self.base_overlap = base_overlap
        self.height_variance_compensation = height_variance_compensation
    
    def calculate_layer_overlaps(
        self,
        layer_heights: List[float],
        reference_distance: float,
    ) -> List[float]:
        """Calculate overlap adjustments for each layer.
        
        Args:
            layer_heights: List of layer heights above ground
            reference_distance: Reference capture distance
        
        Returns:
            List of adjusted overlap ratios for each layer
        """
        overlaps = []
        
        for height in layer_heights:
            # Distance increases with height (simplified model)
            # In reality, this depends on building geometry
            distance_ratio = max(1.0, height / max(reference_distance, 1.0))
            
            # Adjust overlap to compensate
            adjusted_overlap = self.base_overlap * (1 + self.height_variance_compensation * (distance_ratio - 1))
            adjusted_overlap = min(0.9, adjusted_overlap)  # Cap at 90%
            
            overlaps.append(adjusted_overlap)
        
        return overlaps
    
    def optimize_transition_overlap(
        self,
        upper_layer_overlap: float,
        lower_layer_overlap: float,
    ) -> float:
        """Optimize overlap at layer transitions.
        
        Args:
            upper_layer_overlap: Overlap for upper layer
            lower_layer_overlap: Overlap for lower layer
        
        Returns:
            Optimized transition overlap
        """
        # Smooth transition by averaging
        return (upper_layer_overlap + lower_layer_overlap) / 2


class CaptureDistanceEnsurer:
    """Ensures consistent capture distance across coverage area.
    
    This is critical for maintaining consistent image scale
    and overlap for 3D reconstruction.
    """
    
    def __init__(
        self,
        target_distance: float,
        tolerance: float = 0.15,
    ):
        """Initialize capture distance ensurer.
        
        Args:
            target_distance: Target capture distance in meters
            tolerance: Tolerance ratio for distance variance
        """
        self.target_distance = target_distance
        self.tolerance = tolerance
    
    def check_distance_consistency(
        self,
        distances: List[float],
    ) -> Tuple[bool, dict]:
        """Check if capture distances are consistent.
        
        Args:
            distances: List of capture distances
        
        Returns:
            Tuple of (is_consistent, statistics dict)
        """
        if not distances:
            return True, {"mean": 0, "variance": 0, "out_of_tolerance": 0}
        
        mean_dist = np.mean(distances)
        variance = np.var(distances)
        
        # Calculate out-of-tolerance count
        lower_bound = self.target_distance * (1 - self.tolerance)
        upper_bound = self.target_distance * (1 + self.tolerance)
        
        out_of_tolerance = sum(
            1 for d in distances 
            if d < lower_bound or d > upper_bound
        )
        
        is_consistent = out_of_tolerance == 0
        
        return is_consistent, {
            "mean": mean_dist,
            "variance": variance,
            "min": np.min(distances),
            "max": np.max(distances),
            "out_of_tolerance": out_of_tolerance,
            "total": len(distances),
        }
    
    def suggest_distance_adjustments(
        self,
        waypoints: List[np.ndarray],
        surface_points: List[np.ndarray],
    ) -> List[float]:
        """Suggest distance adjustments for waypoints.
        
        Args:
            waypoints: List of waypoint positions
            surface_points: Corresponding surface contact points
        
        Returns:
            List of suggested distance adjustments (in meters)
        """
        adjustments = []
        
        for wp, sp in zip(waypoints, surface_points):
            current_distance = np.linalg.norm(wp - sp)
            adjustment = self.target_distance - current_distance
            adjustments.append(adjustment)
        
        return adjustments
    
    def create_distance_uniform_path(
        self,
        path_points: np.ndarray,
        distances: List[float],
    ) -> np.ndarray:
        """Create path with uniform capture distance.
        
        Args:
            path_points: Original path points
            distances: Capture distances at each point
        
        Returns:
            Adjusted path with uniform distance
        """
        if len(path_points) != len(distances):
            raise ValueError("Path points and distances must have same length")
        
        adjusted = np.zeros_like(path_points)
        
        for i, (point, dist) in enumerate(zip(path_points, distances)):
            if dist > 0:
                # Move point to achieve target distance
                # Direction is perpendicular to path (z-axis in 3D space)
                direction = np.array([0, 0, 1])  # Upward direction
                adjusted[i] = point + direction * (self.target_distance - dist)
            else:
                adjusted[i] = point
        
        return adjusted
