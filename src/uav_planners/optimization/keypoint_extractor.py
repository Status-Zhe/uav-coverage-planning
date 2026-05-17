"""Keypoint extractor for marking critical waypoints.

Identifies which waypoints are essential (keypoints) vs interpolated.
"""

from typing import List
import numpy as np

from ..models.waypoint import Waypoint


class KeypointExtractor:
    """Extracts and marks keypoints in a trajectory.
    
    Keypoints are:
    - Turn points (significant heading change)
    - Photo points (where images are captured)
    - Altitude change points
    - Start and end points
    """
    
    def __init__(
        self,
        min_heading_change: float = 15.0,  # degrees
        min_altitude_change: float = 2.0,  # meters
        min_distance: float = 5.0  # meters between keypoints
    ):
        """Initialize keypoint extractor.
        
        Args:
            min_heading_change: Minimum heading change to mark as keypoint
            min_altitude_change: Minimum altitude change to mark as keypoint
            min_distance: Minimum distance between keypoints
        """
        self.min_heading_change = min_heading_change
        self.min_altitude_change = min_altitude_change
        self.min_distance = min_distance
    
    def extract_keypoints(
        self,
        waypoints: List[Waypoint],
        mark_original: bool = True
    ) -> List[Waypoint]:
        """Extract keypoints from waypoint list.
        
        Args:
            waypoints: Input waypoints
            mark_original: If True, mark all input waypoints as keypoints
            
        Returns:
            Waypoints with is_keypoint marked
        """
        if not waypoints:
            return []
        
        if mark_original:
            # Mark all input waypoints as keypoints
            for wp in waypoints:
                wp.is_keypoint = True
            return waypoints
        
        # Otherwise, detect keypoints based on geometry
        keypoint_indices = self._detect_keypoint_indices(waypoints)
        
        # Mark keypoints
        for i, wp in enumerate(waypoints):
            wp.is_keypoint = i in keypoint_indices
        
        return waypoints
    
    def _detect_keypoint_indices(self, waypoints: List[Waypoint]) -> set:
        """Detect which indices should be keypoints.
        
        Args:
            waypoints: Input waypoints
            
        Returns:
            Set of keypoint indices
        """
        if len(waypoints) < 2:
            return set(range(len(waypoints)))
        
        keypoints = {0, len(waypoints) - 1}  # Always include start and end
        
        for i in range(1, len(waypoints) - 1):
            prev = waypoints[i - 1]
            curr = waypoints[i]
            next_wp = waypoints[i + 1]
            
            # Check heading change
            heading1 = np.degrees(np.arctan2(curr.y - prev.y, curr.x - prev.x))
            heading2 = np.degrees(np.arctan2(next_wp.y - curr.y, next_wp.x - curr.x))
            heading_change = abs(((heading2 - heading1 + 180) % 360) - 180)
            
            if heading_change > self.min_heading_change:
                keypoints.add(i)
                continue
            
            # Check altitude change
            altitude_change = abs(next_wp.z - curr.z)
            if altitude_change > self.min_altitude_change:
                keypoints.add(i)
                continue
            
            # Check distance from last keypoint
            if keypoints:
                last_kp_idx = max(idx for idx in keypoints if idx < i)
                last_kp = waypoints[last_kp_idx]
                dist = np.sqrt(
                    (curr.x - last_kp.x)**2 +
                    (curr.y - last_kp.y)**2 +
                    (curr.z - last_kp.z)**2
                )
                if dist > self.min_distance * 3:  # Force keypoint if too far
                    keypoints.add(i)
        
        return keypoints
    
    def separate_keypoints_and_trajectory(
        self,
        waypoints: List[Waypoint]
    ) -> tuple:
        """Separate waypoints into keypoints and interpolated trajectory.
        
        Args:
            waypoints: Mixed waypoint list
            
        Returns:
            Tuple of (keypoints, interpolated_trajectory)
        """
        keypoints = [wp for wp in waypoints if wp.is_keypoint]
        trajectory = [wp for wp in waypoints if not wp.is_keypoint]
        
        return keypoints, trajectory
