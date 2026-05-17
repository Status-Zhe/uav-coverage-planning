"""Point cloud preprocessing utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..models.pointcloud import PointCloud


def _voxel_keys(points: np.ndarray, voxel_size: float, min_point: np.ndarray) -> np.ndarray:
    scaled = np.floor((points - min_point.reshape(1, 3)) / voxel_size).astype(np.int64)
    return scaled


def _voxel_downsample_indices(points: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0 or voxel_size <= 0:
        indices = np.arange(points.shape[0], dtype=np.int64)
        return indices, np.zeros(3, dtype=np.float64)

    min_point = np.min(points, axis=0)
    keys = _voxel_keys(points, voxel_size, min_point)
    key_view = np.ascontiguousarray(keys).view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1])))
    _, first_indices = np.unique(key_view, return_index=True)
    first_indices = np.sort(first_indices)
    return first_indices.astype(np.int64), min_point


def _statistical_mask(points: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    count = points.shape[0]
    if count <= 3 or k <= 0:
        return np.ones(count, dtype=bool)

    k_eff = min(k, count - 1)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k_eff + 1)
    mean_dist = np.mean(distances[:, 1:], axis=1)

    threshold = float(np.mean(mean_dist) + std_ratio * np.std(mean_dist))
    return mean_dist <= threshold


def _radius_mask(points: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    count = points.shape[0]
    if count <= 3 or radius <= 0 or min_neighbors <= 0:
        return np.ones(count, dtype=bool)

    tree = cKDTree(points)
    neighbor_lists = tree.query_ball_point(points, r=radius)
    neighbor_count = np.array([max(0, len(indices) - 1) for indices in neighbor_lists], dtype=np.int32)
    return neighbor_count >= int(min_neighbors)


def _largest_cluster_mask(
    points: np.ndarray,
    eps: float,
    min_samples: int,
    max_points: int,
) -> np.ndarray:
    count = points.shape[0]
    if count == 0:
        return np.zeros(0, dtype=bool)
    if count < max(3, min_samples):
        return np.ones(count, dtype=bool)

    from sklearn.cluster import DBSCAN

    if count <= max_points:
        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(points)
        valid = labels >= 0
        if not np.any(valid):
            return np.ones(count, dtype=bool)
        unique_labels, counts = np.unique(labels[valid], return_counts=True)
        largest = unique_labels[int(np.argmax(counts))]
        return labels == largest

    rng = np.random.default_rng(42)
    sampled_idx = rng.choice(count, size=max_points, replace=False)
    sampled_points = points[sampled_idx]
    sampled_labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(sampled_points)

    valid = sampled_labels >= 0
    if not np.any(valid):
        return np.ones(count, dtype=bool)

    unique_labels, counts = np.unique(sampled_labels[valid], return_counts=True)
    largest = unique_labels[int(np.argmax(counts))]
    largest_points = sampled_points[sampled_labels == largest]

    if largest_points.shape[0] == 0:
        return np.ones(count, dtype=bool)

    tree = cKDTree(largest_points)
    distances, _ = tree.query(points, k=1)
    return distances <= (eps * 1.5)


def preprocess_pointcloud(
    pointcloud: PointCloud,
    enable: bool = True,
    voxel_size_m: float = 1.5,
    statistical_k: int = 24,
    statistical_std_ratio: float = 2.0,
    radius_m: float = 4,
    radius_min_neighbors: int = 4,
    keep_largest_cluster: bool = True,
    cluster_eps_m: float = 3.0,
    cluster_min_samples: int = 10,
    cluster_max_points: int = 120000,
    min_remaining_points: int = 2,
) -> PointCloud:
    if not enable:
        return pointcloud

    points = pointcloud.points
    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty before preprocessing")

    sampled_idx, min_point = _voxel_downsample_indices(points, voxel_size_m)
    sampled_points = points[sampled_idx]

    mask_stat = _statistical_mask(sampled_points, k=statistical_k, std_ratio=statistical_std_ratio)
    filtered_points = sampled_points[mask_stat]

    if filtered_points.shape[0] == 0:
        raise ValueError("Point cloud became empty after statistical outlier filtering")

    mask_radius = _radius_mask(filtered_points, radius=radius_m, min_neighbors=radius_min_neighbors)
    filtered_points = filtered_points[mask_radius]

    if filtered_points.shape[0] == 0:
        raise ValueError("Point cloud became empty after radius outlier filtering")

    if keep_largest_cluster:
        mask_cluster = _largest_cluster_mask(
            filtered_points,
            eps=cluster_eps_m,
            min_samples=cluster_min_samples,
            max_points=cluster_max_points,
        )
        filtered_points = filtered_points[mask_cluster]
        if filtered_points.shape[0] == 0:
            raise ValueError("Point cloud became empty after largest-cluster extraction")

    if voxel_size_m > 0:
        all_keys = _voxel_keys(points, voxel_size_m, min_point)
        keep_keys = _voxel_keys(filtered_points, voxel_size_m, min_point)
        all_view = np.ascontiguousarray(all_keys).view(np.dtype((np.void, all_keys.dtype.itemsize * all_keys.shape[1])))
        keep_view = np.ascontiguousarray(keep_keys).view(np.dtype((np.void, keep_keys.dtype.itemsize * keep_keys.shape[1])))
        final_mask = np.isin(all_view, keep_view).reshape(-1)
    else:
        tree = cKDTree(filtered_points)
        dist, _ = tree.query(points, k=1)
        final_mask = (dist <= max(1e-6, cluster_eps_m)).reshape(-1)

    kept_count = int(np.sum(final_mask))
    if kept_count < int(min_remaining_points):
        raise ValueError(
            f"Point cloud has too few points after preprocessing: {kept_count} < {int(min_remaining_points)}"
        )

    result_points = points[final_mask]
    result_normals: Optional[np.ndarray] = None
    result_colors: Optional[np.ndarray] = None

    if pointcloud.normals is not None and pointcloud.normals.shape[0] == points.shape[0]:
        result_normals = pointcloud.normals[final_mask]

    if pointcloud.colors is not None and pointcloud.colors.shape[0] == points.shape[0]:
        result_colors = pointcloud.colors[final_mask]

    return PointCloud(
        points=result_points,
        normals=result_normals,
        colors=result_colors,
        source_file=pointcloud.source_file,
        coordinate_frame=pointcloud.coordinate_frame,
        enu_origin_ecef=pointcloud.enu_origin_ecef,
    )
