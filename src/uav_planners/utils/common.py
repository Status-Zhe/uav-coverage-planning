"""Common utility functions for UAV coverage planning.

This module provides shared utilities across all components,
ensuring DRY (Don't Repeat Yourself) principles.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Any, Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Geometry Utilities
# =============================================================================

def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        p1: First point (3D)
        p2: Second point (3D)
    
    Returns:
        Distance in meters
    """
    return float(np.linalg.norm(p1 - p2))


def calculate_path_length(waypoints: List[np.ndarray]) -> float:
    """Calculate total path length.
    
    Args:
        waypoints: List of 3D points
    
    Returns:
        Total path length in meters
    """
    if len(waypoints) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(waypoints) - 1):
        total += calculate_distance(waypoints[i], waypoints[i + 1])
    
    return total


def calculate_bearing(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate bearing (heading) from p1 to p2 in degrees.
    
    Args:
        p1: Start point (x, y, z)
        p2: End point (x, y, z)
    
    Returns:
        Bearing in degrees (0-360, 0=North, 90=East)
    """
    dx = p2[0] - p1[0]  # East
    dy = p2[1] - p1[1]  # North
    
    bearing = np.degrees(np.arctan2(dx, dy))  # Note: x=East, y=North
    if bearing < 0:
        bearing += 360
    
    return float(bearing)


def normalize_angle(angle_deg: float) -> float:
    """Normalize angle to [0, 360) range.
    
    Args:
        angle_deg: Angle in degrees
    
    Returns:
        Normalized angle
    """
    angle = angle_deg % 360
    if angle < 0:
        angle += 360
    return float(angle)


def interpolate_points(
    p1: np.ndarray,
    p2: np.ndarray,
    num_points: int,
) -> List[np.ndarray]:
    """Interpolate points between p1 and p2.
    
    Args:
        p1: Start point
        p2: End point
        num_points: Number of points to generate (including endpoints)
    
    Returns:
        List of interpolated points
    """
    if num_points < 2:
        return [p1]
    
    t = np.linspace(0, 1, num_points)
    result = []
    
    for ti in t:
        point = p1 + ti * (p2 - p1)
        result.append(point)
    
    return result


def project_to_2d(point: np.ndarray) -> Tuple[float, float]:
    """Project 3D point to 2D (x, y) plane.
    
    Args:
        point: 3D point
    
    Returns:
        Tuple of (x, y)
    """
    return float(point[0]), float(point[1])


def get_bounding_box(
    points: List[np.ndarray],
) -> Tuple[float, float, float, float]:
    """Get axis-aligned bounding box of points.
    
    Args:
        points: List of 3D points
    
    Returns:
        Tuple of (xmin, ymin, xmax, ymax)
    """
    if not points:
        return (0, 0, 0, 0)
    
    points_array = np.array(points)
    xmin, ymin = np.min(points_array[:, :2], axis=0)
    xmax, ymax = np.max(points_array[:, :2], axis=0)
    
    return (float(xmin), float(ymin), float(xmax), float(ymax))


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive.
    
    Args:
        value: Value to validate
        name: Name for error message
    
    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_range(
    value: float,
    name: str,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> None:
    """Validate that a value is within range.
    
    Args:
        value: Value to validate
        name: Name for error message
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Raises:
        ValueError: If value is out of range
    """
    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")


def validate_overlap(value: float, name: str = "overlap") -> None:
    """Validate overlap ratio.
    
    Args:
        value: Overlap ratio to validate
        name: Name for error message
    
    Raises:
        ValueError: If overlap is not in [0, 1)
    """
    if value < 0 or value >= 1:
        raise ValueError(f"{name} must be in [0, 1), got {value}")


def validate_not_empty(collection: List, name: str) -> None:
    """Validate that a collection is not empty.
    
    Args:
        collection: Collection to validate
        name: Name for error message
    
    Raises:
        ValueError: If collection is empty
    """
    if not collection:
        raise ValueError(f"{name} cannot be empty")


# =============================================================================
# List Utilities
# =============================================================================

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def batch_items(
    items: List[Any],
    batch_size: int,
) -> List[List[Any]]:
    """Batch items for processing.
    
    Args:
        items: Items to batch
        batch_size: Size of each batch
    
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def remove_duplicates(
    items: List,
    key: Optional[Callable[[Any], Any]] = None,
) -> List:
    """Remove duplicates while preserving order.
    
    Args:
        items: List of items
        key: Optional key function for comparison
    
    Returns:
        List with duplicates removed
    """
    if key is None:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    else:
        seen = set()
        result = []
        for item in items:
            k = key(item)
            if k not in seen:
                seen.add(k)
                result.append(item)
        return result


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide with default for zero denominator.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
    
    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


# =============================================================================
# Coordinate Transformations
# =============================================================================

def local_to_enu(
    local_point: np.ndarray,
    origin: np.ndarray,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Transform local coordinates to ENU.
    
    Args:
        local_point: Point in local coordinates
        origin: Origin of local frame in ENU
        yaw_deg: Yaw rotation in degrees
    
    Returns:
        Point in ENU coordinates
    """
    # Rotation matrix
    yaw_rad = np.radians(yaw_deg)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    
    rotation = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1],
    ])
    
    rotated = rotation @ local_point
    return rotated + origin


def enu_to_local(
    enu_point: np.ndarray,
    origin: np.ndarray,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Transform ENU coordinates to local.
    
    Args:
        enu_point: Point in ENU coordinates
        origin: Origin of local frame in ENU
        yaw_deg: Yaw rotation in degrees
    
    Returns:
        Point in local coordinates
    """
    # Translate to origin
    translated = enu_point - origin
    
    # Inverse rotation matrix
    yaw_rad = np.radians(yaw_deg)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    
    rotation_inv = np.array([
        [cos_yaw, sin_yaw, 0],
        [-sin_yaw, cos_yaw, 0],
        [0, 0, 1],
    ])
    
    return rotation_inv @ translated


# =============================================================================
# Math Utilities
# =============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b.
    
    Args:
        a: Start value
        b: End value
        t: Interpolation factor [0, 1]
    
    Returns:
        Interpolated value
    """
    return a + t * (b - a)


def smoothstep(t: float) -> float:
    """Smoothstep function (ease in/out).
    
    Args:
        t: Input value [0, 1]
    
    Returns:
        Smoothed value [0, 1]
    """
    t = clamp(t, 0, 1)
    return t * t * (3 - 2 * t)


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return float(np.radians(degrees))


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return float(np.degrees(radians))


# =============================================================================
# Statistics Utilities
# =============================================================================

def calculate_statistics(values: List[float]) -> dict:
    """Calculate basic statistics for a list of values.
    
    Args:
        values: List of numeric values
    
    Returns:
        Dictionary with mean, std, min, max
    """
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    
    values_array = np.array(values)
    
    return {
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array)),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "median": float(np.median(values_array)),
    }


def moving_average(values: List[float], window: int) -> List[float]:
    """Calculate moving average.
    
    Args:
        values: List of values
        window: Window size
    
    Returns:
        List of smoothed values
    """
    if window < 1:
        return values
    
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_values = values[start:i + 1]
        result.append(sum(window_values) / len(window_values))
    
    return result
