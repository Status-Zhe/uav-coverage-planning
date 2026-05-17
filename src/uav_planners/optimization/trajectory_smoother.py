"""Trajectory smoother with B-Spline interpolation.

Converts sparse keypoints to dense interpolated trajectory.
Supports different interpolation methods and interval control.
"""

from typing import List, Tuple
import numpy as np
from scipy import interpolate

from ..models.waypoint import Waypoint, WaypointAction


class TrajectorySmoother:
    """B-Spline trajectory smoother.
    
    Input: Sparse keypoints (is_keypoint=True)
    Output: Dense interpolated trajectory (is_keypoint=False)
    
    Features:
    - B-Spline smoothing for continuous trajectory
    - Configurable interpolation interval
    - Preserves heading and gimbal angles through interpolation
    - Maintains speed and action attributes
    """
    
    def __init__(self, interpolation_interval: float = 0.5, smoothing_factor: float = 0.0):
        """Initialize trajectory smoother.
        
        Args:
            interpolation_interval: Distance between interpolated points (meters)
            smoothing_factor: B-Spline smoothing factor (0.0 = exact interpolation)
        """
        self.interpolation_interval = interpolation_interval
        self.smoothing_factor = smoothing_factor
    
    def smooth(
        self,
        waypoints: List[Waypoint],
        output_keypoints: bool = True
    ) -> List[Waypoint]:
        """Smooth trajectory using B-Spline interpolation.
        
        Args:
            waypoints: Input waypoints (should include keypoints)
            output_keypoints: If True, include original keypoints in output
            
        Returns:
            Dense trajectory with interpolated points
        """
        if len(waypoints) < 2:
            return waypoints
        
        if len(waypoints) < 4:
            # Not enough points for B-Spline, use linear interpolation
            return self._linear_interpolate(waypoints, output_keypoints)
        
        # Extract positions
        positions = np.array([[wp.x, wp.y, wp.z] for wp in waypoints])
        
        # Calculate cumulative distance along path (parameter t)
        distances = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            distances[i] = distances[i-1] + np.linalg.norm(positions[i] - positions[i-1])
        
        total_distance = distances[-1]
        
        if total_distance < self.interpolation_interval:
            # Path too short, return original
            return waypoints
        
        # Normalize to [0, 1]
        t = distances / total_distance
        
        # Create B-Spline representation (degree 3 = cubic)
        try:
            # Separate spline for each dimension
            spline_x = interpolate.make_interp_spline(t, positions[:, 0], k=3)
            spline_y = interpolate.make_interp_spline(t, positions[:, 1], k=3)
            spline_z = interpolate.make_interp_spline(t, positions[:, 2], k=3)
        except ValueError:
            # Fall back to linear if B-Spline fails
            return self._linear_interpolate(waypoints, output_keypoints)
        
        # Generate dense samples
        num_points = max(int(total_distance / self.interpolation_interval), len(waypoints))
        t_dense = np.linspace(0, 1, num_points)
        
        # Evaluate splines
        x_dense = spline_x(t_dense)
        y_dense = spline_y(t_dense)
        z_dense = spline_z(t_dense)
        
        # Interpolate other attributes
        headings = np.array([wp.heading_deg for wp in waypoints])
        pitches = np.array([wp.gimbal_pitch_deg for wp in waypoints])
        speeds = np.array([wp.speed_ms for wp in waypoints])
        
        # Use linear interpolation for angles (handle wrap-around)
        heading_dense = self._interpolate_angles(t, headings, t_dense)
        pitch_dense = np.interp(t_dense, t, pitches)
        speed_dense = np.interp(t_dense, t, speeds)
        
        # Build output waypoints
        result = []
        
        # Track which dense points correspond to original keypoints
        keypoint_indices = set()
        if output_keypoints:
            for i, t_orig in enumerate(t):
                idx = np.argmin(np.abs(t_dense - t_orig))
                keypoint_indices.add(idx)
        
        for i in range(len(t_dense)):
            is_kp = i in keypoint_indices
            
            # Determine action: shoot if keypoint, hover otherwise
            action = WaypointAction.SHOOT if is_kp else WaypointAction.HOVER
            
            wp = Waypoint(
                x=float(x_dense[i]),
                y=float(y_dense[i]),
                z=float(z_dense[i]),
                heading_deg=float(heading_dense[i]),
                gimbal_pitch_deg=float(pitch_dense[i]),
                speed_ms=float(speed_dense[i]),
                action=action,
                dwell_time_s=0.5 if is_kp else 0.0,  # Longer dwell at keypoints
                is_keypoint=is_kp
            )
            result.append(wp)
        
        return result
    
    def _linear_interpolate(
        self,
        waypoints: List[Waypoint],
        output_keypoints: bool
    ) -> List[Waypoint]:
        """Linear interpolation fallback.
        
        Args:
            waypoints: Input waypoints
            output_keypoints: Whether to mark original points as keypoints
            
        Returns:
            Linearly interpolated trajectory
        """
        result = []
        
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            
            # Calculate segment length
            dist = np.sqrt(
                (wp2.x - wp1.x)**2 +
                (wp2.y - wp1.y)**2 +
                (wp2.z - wp1.z)**2
            )
            
            # Number of points for this segment
            num_points = max(2, int(dist / self.interpolation_interval))
            
            for j in range(num_points):
                t = j / (num_points - 1) if num_points > 1 else 0
                
                # Linear interpolation
                x = wp1.x + t * (wp2.x - wp1.x)
                y = wp1.y + t * (wp2.y - wp1.y)
                z = wp1.z + t * (wp2.z - wp1.z)
                
                # Angle interpolation (handle wrap-around)
                heading = self._lerp_angle(wp1.heading_deg, wp2.heading_deg, t)
                pitch = wp1.gimbal_pitch_deg + t * (wp2.gimbal_pitch_deg - wp1.gimbal_pitch_deg)
                speed = wp1.speed_ms + t * (wp2.speed_ms - wp1.speed_ms)
                
                # Mark as keypoint if it's an original waypoint
                is_kp = output_keypoints and (j == 0 or (i == len(waypoints) - 2 and j == num_points - 1))
                action = WaypointAction.SHOOT if is_kp else WaypointAction.HOVER
                
                wp = Waypoint(
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    heading_deg=float(heading),
                    gimbal_pitch_deg=float(pitch),
                    speed_ms=float(speed),
                    action=action,
                    dwell_time_s=0.5 if is_kp else 0.0,
                    is_keypoint=is_kp
                )
                result.append(wp)
        
        # Remove duplicates
        if result:
            filtered = [result[0]]
            for wp in result[1:]:
                prev = filtered[-1]
                dist = np.sqrt((wp.x - prev.x)**2 + (wp.y - prev.y)**2 + (wp.z - prev.z)**2)
                if dist > 0.01:  # 1cm threshold
                    filtered.append(wp)
            result = filtered
        
        return result
    
    def _interpolate_angles(
        self,
        t_orig: np.ndarray,
        angles: np.ndarray,
        t_new: np.ndarray
    ) -> np.ndarray:
        """Interpolate angles handling wrap-around at 360 degrees.
        
        Args:
            t_orig: Original parameter values
            angles: Original angles in degrees
            t_new: New parameter values
            
        Returns:
            Interpolated angles
        """
        # Convert to complex numbers to handle wrap-around
        complex_angles = np.exp(1j * np.radians(angles))
        
        # Interpolate real and imaginary parts separately
        real_interp = np.interp(t_new, t_orig, complex_angles.real)
        imag_interp = np.interp(t_new, t_orig, complex_angles.imag)
        
        # Convert back to angles
        result = np.degrees(np.arctan2(imag_interp, real_interp))
        
        return result
    
    def _lerp_angle(self, a1: float, a2: float, t: float) -> float:
        """Linear interpolation between two angles handling wrap-around.
        
        Args:
            a1: Start angle in degrees
            a2: End angle in degrees
            t: Interpolation factor [0, 1]
            
        Returns:
            Interpolated angle
        """
        # Normalize angles to [-180, 180]
        a1 = ((a1 + 180) % 360) - 180
        a2 = ((a2 + 180) % 360) - 180
        
        # Find shortest path
        diff = a2 - a1
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        
        result = a1 + t * diff
        return ((result + 360) % 360)  # Normalize to [0, 360)
    
    def smooth_with_constraints(
        self,
        waypoints: List[Waypoint],
        min_segment_length: float = 1.0,
        max_curvature: float = 30.0  # degrees per meter
    ) -> List[Waypoint]:
        """Smooth trajectory with curvature constraints.
        
        Ensures that the smoothed trajectory doesn't exceed maximum curvature,
        which is important for UAV flight dynamics.
        
        Args:
            waypoints: Input waypoints
            min_segment_length: Minimum distance between points
            max_curvature: Maximum allowed heading change per meter
            
        Returns:
            Constrained smoothed trajectory
        """
        # First do normal smoothing
        smoothed = self.smooth(waypoints, output_keypoints=True)
        
        if len(smoothed) < 3:
            return smoothed
        
        # Check curvature constraints
        result = [smoothed[0]]
        
        for i in range(1, len(smoothed) - 1):
            prev = result[-1]
            curr = smoothed[i]
            next_wp = smoothed[i + 1]
            
            # Calculate segment lengths
            dist1 = np.sqrt(
                (curr.x - prev.x)**2 +
                (curr.y - prev.y)**2 +
                (curr.z - prev.z)**2
            )
            dist2 = np.sqrt(
                (next_wp.x - curr.x)**2 +
                (next_wp.y - curr.y)**2 +
                (next_wp.z - curr.z)**2
            )
            
            # Calculate heading change
            heading1 = np.degrees(np.arctan2(curr.y - prev.y, curr.x - prev.x))
            heading2 = np.degrees(np.arctan2(next_wp.y - curr.y, next_wp.x - curr.x))
            heading_change = abs(((heading2 - heading1 + 180) % 360) - 180)
            
            # Check constraints
            avg_dist = (dist1 + dist2) / 2
            curvature = heading_change / avg_dist if avg_dist > 0 else 0
            
            if curvature > max_curvature or dist1 < min_segment_length:
                # Skip this point if constraints violated
                continue
            
            result.append(curr)
        
        result.append(smoothed[-1])
        
        return result
