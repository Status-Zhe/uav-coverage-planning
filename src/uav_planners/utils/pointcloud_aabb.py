"""PointCloud AABB utilities."""

from typing import Optional, Tuple

import numpy as np

from ..models.pointcloud import PointCloud


def xy_rotation_center_to_transform_matrix(
    yaw_deg: float,
    center: Tuple[float, float, float],
) -> np.ndarray:
    """Build a 4x4 transform matrix from XY-plane yaw and center translation.

    The returned matrix maps box-local coordinates to world coordinates.
    """
    center_arr = np.asarray(center, dtype=np.float64)
    if center_arr.shape != (3,):
        raise ValueError("center must be a 3-element tuple")

    yaw_rad = np.deg2rad(float(yaw_deg))
    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))

    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = c
    transform[0, 1] = -s
    transform[1, 0] = s
    transform[1, 1] = c
    transform[:3, 3] = center_arr
    return transform


def _normalize_transform_matrix(transform_matrix: np.ndarray | Tuple) -> np.ndarray:
    matrix = np.asarray(transform_matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("aabb_transform must be a 4x4 matrix")
    return matrix


def _normalize_size(size: Tuple[float, float, float]) -> np.ndarray:
    size_arr = np.asarray(size, dtype=np.float64)
    if size_arr.shape != (3,):
        raise ValueError("AABB size must be a 3-element tuple")
    if np.any(size_arr <= 0):
        raise ValueError("AABB size values must all be > 0")
    return size_arr


def crop_pointcloud_with_transform_box(
    pointcloud: PointCloud,
    transform_matrix: np.ndarray | Tuple,
    size: Tuple[float, float, float],
) -> Tuple[PointCloud, np.ndarray, np.ndarray, int, int]:
    """Crop point cloud with oriented box (OBB) defined by transform + size.

    Args:
        pointcloud: Source point cloud in world coordinates.
        transform_matrix: 4x4 matrix mapping box local -> world.
        size: Box lengths along local x/y/z axes.

    Returns:
        (cropped_pointcloud, local_min, local_max, before_count, after_count)
    """
    transform = _normalize_transform_matrix(transform_matrix)
    size_arr = _normalize_size(size)
    half = size_arr / 2.0

    local_min = -half
    local_max = half

    inv_transform = np.linalg.inv(transform)
    points = pointcloud.points
    before_count = int(points.shape[0])

    ones = np.ones((before_count, 1), dtype=np.float64)
    points_h = np.hstack([points.astype(np.float64), ones])
    local_points_h = (inv_transform @ points_h.T).T
    local_points = local_points_h[:, :3]

    mask = np.all((local_points >= local_min) & (local_points <= local_max), axis=1)
    after_count = int(np.count_nonzero(mask))
    if after_count == 0:
        raise ValueError("Transform box contains 0 points, please adjust transform/size")

    normals = None
    if pointcloud.normals is not None and pointcloud.normals.shape[0] == before_count:
        normals = pointcloud.normals[mask]

    colors = None
    if pointcloud.colors is not None and pointcloud.colors.shape[0] == before_count:
        colors = pointcloud.colors[mask]

    cropped = PointCloud(
        points=points[mask],
        normals=normals,
        colors=colors,
        source_file=pointcloud.source_file,
        coordinate_frame=pointcloud.coordinate_frame,
        enu_origin_ecef=pointcloud.enu_origin_ecef,
    )
    return cropped, local_min, local_max, before_count, after_count


def aabb_min_max_from_center_size(
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute AABB min/max corners from center and size."""
    center_arr = np.asarray(center, dtype=np.float64)
    size_arr = np.asarray(size, dtype=np.float64)
    if center_arr.shape != (3,) or size_arr.shape != (3,):
        raise ValueError("AABB center and size must be 3-element tuples")
    if np.any(size_arr <= 0):
        raise ValueError("AABB size values must all be > 0")

    half = size_arr / 2.0
    return center_arr - half, center_arr + half


def crop_pointcloud_with_aabb(
    pointcloud: PointCloud,
    center: Optional[Tuple[float, float, float]],
    size: Tuple[float, float, float],
) -> Tuple[PointCloud, np.ndarray, np.ndarray, int, int]:
    """Crop a PointCloud with axis-aligned box.

    Returns:
        (cropped_pointcloud, min_xyz, max_xyz, before_count, after_count)
    """
    if center is None:
        center = pointcloud.bounds.center

    transform = xy_rotation_center_to_transform_matrix(0.0, center)
    return crop_pointcloud_with_transform_box(pointcloud, transform, size)
