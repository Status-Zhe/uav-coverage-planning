"""Coordinate system utilities for adaptive point cloud handling.

Automatically detects and handles different coordinate systems:
- Camera coords: Z negative, pointing down (common in photogrammetry)
- Local/ENU coords: Z positive, pointing up (standard for planning)
"""

import numpy as np
from typing import Tuple
from enum import Enum


class CoordinateSystem(Enum):
    """Supported coordinate systems."""
    CAMERA = "camera"  # Z negative, down from camera
    LOCAL = "local"    # Z positive, up from ground
    UNKNOWN = "unknown"


def detect_coordinate_system(points: np.ndarray) -> Tuple[CoordinateSystem, float, str]:
    """Detect coordinate system from point cloud Z values.
    
    Heuristics:
    - If max Z < 0: Camera coordinate system
    - If min Z >= 0: Local coordinate system  
    - Mixed: Unknown (assume local)
    
    Args:
        points: Nx3 numpy array of points
        
    Returns:
        Tuple of (coord_system, z_offset, explanation)
        - coord_system: Detected coordinate system
        - z_offset: Amount to shift Z to convert to local (ground=0)
        - explanation: Human-readable description
    """
    z_min = points[:, 2].min()
    z_max = points[:, 2].max()
    
    # Camera coordinate system: Z is negative, pointing down from camera
    # Ground is at some negative Z value
    if z_max < 0 or (z_max < 10 and z_min < -10):
        # Shift so min Z = 0 (ground level)
        z_offset = -z_min
        return (
            CoordinateSystem.CAMERA,
            z_offset,
            f"Camera coords detected (Z={z_min:.1f} to {z_max:.1f}), offset by {z_offset:.1f}m"
        )
    
    # Local coordinate system: Z is positive, ground at or near 0
    elif z_min >= -1:  # Allow small negative values near ground
        return (
            CoordinateSystem.LOCAL,
            0.0,
            f"Local coords detected (Z={z_min:.1f} to {z_max:.1f}), no offset needed"
        )
    
    # Mixed or unknown - assume local with offset
    else:
        z_offset = -z_min if z_min < 0 else 0.0
        return (
            CoordinateSystem.UNKNOWN,
            z_offset,
            f"Unknown coords (Z={z_min:.1f} to {z_max:.1f}), using offset {z_offset:.1f}m"
        )


def to_local_coordinates(points: np.ndarray, z_offset: float) -> np.ndarray:
    """Transform points to local coordinate system (ground at Z=0).
    
    Args:
        points: Nx3 array of points
        z_offset: Amount to add to Z coordinates
        
    Returns:
        Transformed points
    """
    if z_offset == 0:
        return points
    
    points_local = points.copy()
    points_local[:, 2] = points_local[:, 2] + z_offset
    return points_local


def from_local_coordinates(points: np.ndarray, z_offset: float) -> np.ndarray:
    """Transform points from local back to original coordinate system.
    
    Args:
        points: Nx3 array of points in local coords
        z_offset: Original offset amount
        
    Returns:
        Points in original coordinate system
    """
    if z_offset == 0:
        return points
    
    points_original = points.copy()
    points_original[:, 2] = points_original[:, 2] - z_offset
    return points_original


def compute_relative_altitude(
    absolute_altitude: float,
    pointcloud_z_min: float,
    coord_system: CoordinateSystem
) -> float:
    """Compute relative altitude above ground.
    
    For LOCAL coords: altitude is already relative to ground (Z=0)
    For CAMERA coords: altitude should be interpreted as height above ground
    
    Args:
        absolute_altitude: Altitude value from config
        pointcloud_z_min: Minimum Z of point cloud (ground level)
        coord_system: Detected coordinate system
        
    Returns:
        Relative altitude above ground (for use in local coordinate system)
    """
    if coord_system == CoordinateSystem.CAMERA:
        # In camera coords, ground is at negative Z
        # User's altitude is likely intended as height above ground
        # After transformation to local coords, ground is at Z=0
        # So we should use the altitude value directly in local coords
        
        # If user gives small value (< 100), assume it's relative height
        if absolute_altitude < 100:
            return absolute_altitude  # Use directly in local coords
        else:
            # Large value might be absolute, subtract ground level
            return absolute_altitude - abs(pointcloud_z_min)
    else:
        # Local coords - altitude is already relative to ground
        return absolute_altitude


class CoordinateTransformer:
    """Helper class for coordinate transformations."""
    
    def __init__(self, points: np.ndarray):
        """Initialize transformer from point cloud.
        
        Args:
            points: Point cloud points
        """
        self.coord_system, self.z_offset, self.explanation = detect_coordinate_system(points)
        self.z_min = points[:, 2].min()
        self.z_max = points[:, 2].max()
    
    def to_local(self, points: np.ndarray) -> np.ndarray:
        """Transform points to local coordinates."""
        return to_local_coordinates(points, self.z_offset)
    
    def from_local(self, points: np.ndarray) -> np.ndarray:
        """Transform points from local to original coordinates."""
        return from_local_coordinates(points, self.z_offset)
    
    def adjust_altitude(self, altitude: float) -> float:
        """Adjust altitude for coordinate system."""
        return compute_relative_altitude(altitude, self.z_min, self.coord_system)
    
    def is_camera_coords(self) -> bool:
        """Check if camera coordinate system."""
        return self.coord_system == CoordinateSystem.CAMERA
