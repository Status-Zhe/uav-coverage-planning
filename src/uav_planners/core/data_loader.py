"""Data loader module for unified point cloud loading.

This module extracts data loading logic from CoveragePlanner to enable
cleaner separation of concerns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from ..models.pointcloud import PointCloud
from ..io.pointcloud_loader import PointCloudLoader
from ..io.tileset_loader import load_pointcloud_from_tileset
from ..utils.pointcloud_preprocess import preprocess_pointcloud
from ..utils.pointcloud_aabb import xy_rotation_center_to_transform_matrix

logger = logging.getLogger(__name__)


class DataLoader:
    """Unified data loader for UAV coverage planning.
    
    Handles loading from various sources:
    - Point cloud files (.pcd, .ply)
    - 3D Tiles (.json tileset)
    - Synthetic region data (flat surfaces)
    
    Example:
        loader = DataLoader(config)
        target_pc, obstacle_pc = loader.load()
    """
    
    def __init__(self, config: MissionConfig):
        """Initialize data loader.
        
        Args:
            config: Mission configuration with source details
        """
        self.config = config
        self._loader = PointCloudLoader()
    
    def load(self) -> Tuple[PointCloud, PointCloud]:
        """Load target and obstacle point clouds.
        
        Returns:
            Tuple of (target_pointcloud, obstacle_pointcloud)
            For most cases, both are the same.
        """
        if bool(getattr(self.config, "spiral_circle_only_enabled", False)):
            return self._load_spiral_circle_only()

        if bool(getattr(self.config, "cylinder_params_enabled", False)):
            return self._load_cylinder_params()

        if bool(getattr(self.config, "oblique_oneplane_enabled", False)):
            return self._load_oblique_oneplane()

        if bool(getattr(self.config, "region_only_enabled", False)):
            return self._load_region_only()
        
        source_type = getattr(self.config, "data_source_type", "pointcloud_file")
        
        if source_type == "tileset":
            return self._load_tileset()
        else:
            return self._load_pointcloud_file()
    
    def _load_region_only(self) -> Tuple[PointCloud, PointCloud]:
        """Load synthetic point cloud for flat surface planning."""
        rect = getattr(self.config, "coverage_area_rect_xy", None)
        if rect is None:
            raise ValueError("coverage_area_rect_xy is required in region-only mode")
        
        xmin, ymin, xmax, ymax = [float(v) for v in rect]
        ground_z = float(getattr(self.config, "region_ground_z", 0.0))
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        
        synthetic_points = np.array([
            [xmin, ymin, ground_z],
            [xmax, ymin, ground_z],
            [xmax, ymax, ground_z],
            [xmin, ymax, ground_z],
            [center_x, center_y, ground_z],
        ], dtype=float)
        
        synthetic_cloud = PointCloud(points=synthetic_points)
        logger.info(
            "Region-only mode: synthetic pointcloud with %d points",
            synthetic_cloud.point_count,
        )
        return synthetic_cloud, synthetic_cloud

    def _load_spiral_circle_only(self) -> Tuple[PointCloud, PointCloud]:
        """Create a synthetic point cloud for spiral circle-only planning.

        The synthetic points are placed far from the planned spiral so that
        collision checks do not interfere with planning while keeping the
        pipeline data flow intact.
        """
        center_xy = getattr(self.config, "spiral_center_xy", None)
        radius = getattr(self.config, "spiral_radius", None)
        start_z = getattr(self.config, "spiral_start_z", None)
        height = getattr(self.config, "spiral_height", None)

        if center_xy is None or radius is None or start_z is None or height is None:
            raise ValueError("spiral_center_xy, spiral_radius, spiral_start_z, spiral_height are required")

        cx, cy = [float(v) for v in center_xy]
        safe_margin = float(getattr(self.config, "safety_distance", 3.0))
        global_distance = float(getattr(self.config, "global_distance_m", 0.0) or 0.0)
        offset = max(1000.0, float(radius) + global_distance + safe_margin + 10.0)

        far_x = cx + offset
        far_y = cy
        z0 = float(start_z)
        z1 = float(start_z) + float(height)

        synthetic_points = np.array(
            [
                [far_x, far_y, z0],
                [far_x, far_y, z1],
            ],
            dtype=float,
        )

        synthetic_cloud = PointCloud(points=synthetic_points)
        logger.info(
            "Spiral circle-only mode: synthetic pointcloud with %d points",
            synthetic_cloud.point_count,
        )
        return synthetic_cloud, synthetic_cloud

    def _load_cylinder_params(self) -> Tuple[PointCloud, PointCloud]:
        """Create a synthetic point cloud for cylinder parameter planning."""
        center_xy = getattr(self.config, "cylinder_center_xy", None)
        radius = getattr(self.config, "cylinder_radius", None)
        start_z = getattr(self.config, "cylinder_start_z", None)
        height = getattr(self.config, "cylinder_height", None)

        if center_xy is None or radius is None or start_z is None or height is None:
            raise ValueError("cylinder_center_xy, cylinder_radius, cylinder_start_z, cylinder_height are required")

        cx, cy = [float(v) for v in center_xy]
        safe_margin = float(getattr(self.config, "safety_distance", 3.0))
        global_distance = float(getattr(self.config, "global_distance_m", 0.0) or 0.0)
        offset = max(1000.0, float(radius) + global_distance + safe_margin + 10.0)

        far_x = cx + offset
        far_y = cy
        z0 = float(start_z)
        z1 = float(start_z) + float(height)

        synthetic_points = np.array(
            [
                [far_x, far_y, z0],
                [far_x, far_y, z1],
            ],
            dtype=float,
        )

        synthetic_cloud = PointCloud(points=synthetic_points)
        logger.info(
            "Cylinder params mode: synthetic pointcloud with %d points",
            synthetic_cloud.point_count,
        )
        return synthetic_cloud, synthetic_cloud

    def _load_oblique_oneplane(self) -> Tuple[PointCloud, PointCloud]:
        """Create a synthetic point cloud for oblique one-plane planning."""
        polygon_xyz = getattr(self.config, "oneplane_polygon_xyz", None)
        if polygon_xyz is None:
            raise ValueError("oneplane_polygon_xyz is required for oblique one-plane mode")

        polygon_points = np.asarray(polygon_xyz, dtype=np.float64)
        if polygon_points.ndim != 2 or polygon_points.shape[1] != 3:
            raise ValueError("oneplane_polygon_xyz must be a list of 3D points")

        synthetic_cloud = PointCloud(points=polygon_points)
        logger.info(
            "Oblique one-plane mode: synthetic pointcloud with %d points",
            synthetic_cloud.point_count,
        )
        return synthetic_cloud, synthetic_cloud
    
    def _load_tileset(self) -> Tuple[PointCloud, PointCloud]:
        """Load from 3D Tiles tileset."""
        tileset_path = getattr(self.config, "tileset_path", None)
        logger.info(f"Loading 3D Tiles: {tileset_path}")
        
        # Check for conversion mode
        if bool(getattr(self.config, "tiles_convert_to_ply", False)):
            raw_pc = self._convert_and_load_ply()
        else:
            raw_pc = load_pointcloud_from_tileset(
                tileset_path=str(tileset_path),
                tiles_max_points=int(getattr(self.config, "tiles_max_points", 800000)),
                tiles_lod_max=getattr(self.config, "tiles_lod_max", None),
                tiles_bbox=getattr(self.config, "tiles_bbox", None),
                tiles_output_frame=getattr(self.config, "tiles_output_frame", "enu"),
                tiles_input_crs=getattr(self.config, "tiles_input_crs", "auto"),
                tiles_enu_origin_ecef=getattr(self.config, "tiles_enu_origin_ecef", None),
            )
        
        return self._apply_aabb_and_preprocess(raw_pc, raw_pc)
    
    def _convert_and_load_ply(self) -> PointCloud:
        """Convert tileset to PLY then load."""
        from ..io.tileset_loader import export_tileset_to_ply
        
        tileset_path = getattr(self.config, "tileset_path", None)
        converted_ply_path = getattr(self.config, "tiles_converted_ply_path", None)
        
        if not converted_ply_path:
            tileset_file = Path(str(tileset_path))
            converted_ply_path = str(tileset_file.with_name(f"{tileset_file.stem}_converted.ply"))
        
        logger.info(f"Converting tileset to PLY: {converted_ply_path}")
        
        export_tileset_to_ply(
            tileset_path=str(tileset_path),
            output_ply=str(converted_ply_path),
            tiles_max_points=int(getattr(self.config, "tiles_max_points", 800000)),
            tiles_lod_max=getattr(self.config, "tiles_lod_max", None),
            tiles_bbox=getattr(self.config, "tiles_bbox", None),
            write_ascii=bool(getattr(self.config, "tiles_convert_write_ascii", True)),
            coord_frame=str(getattr(self.config, "tiles_convert_coord_frame", "enu")),
            tiles_input_crs=str(getattr(self.config, "tiles_input_crs", "auto")),
            enu_origin_ecef=getattr(self.config, "tiles_enu_origin_ecef", None),
        )
        
        raw_pointcloud = self._loader.load(str(converted_ply_path))
        
        convert_frame = str(getattr(self.config, "tiles_convert_coord_frame", "enu")).lower()
        if convert_frame == "enu":
            raw_pointcloud.coordinate_frame = "enu"
            tiles_origin = getattr(self.config, "tiles_enu_origin_ecef", None)
            if tiles_origin is not None:
                raw_pointcloud.enu_origin_ecef = tuple(float(v) for v in tiles_origin)
        else:
            raw_pointcloud.coordinate_frame = "world"
        
        return raw_pointcloud
    
    def _load_pointcloud_file(self) -> Tuple[PointCloud, PointCloud]:
        """Load from point cloud file (.pcd, .ply)."""
        logger.info(f"Loading point cloud: {self.config.pointcloud_path}")
        raw_pointcloud = self._loader.load(self.config.pointcloud_path)
        
        return self._apply_aabb_and_preprocess(raw_pointcloud, raw_pointcloud)
    
    def _apply_aabb_and_preprocess(
        self, 
        raw_pc: PointCloud, 
        obstacle_pc: PointCloud
    ) -> Tuple[PointCloud, PointCloud]:
        """Apply AABB crop and preprocessing."""
        aabb_size = getattr(self.config, "aabb_size", None)
        aabb_transform = self._get_aabb_transform()
        
        if aabb_size is not None:
            cropped_target = self._loader.apply_optional_aabb(
                raw_pc,
                aabb_size=aabb_size,
                aabb_transform=aabb_transform,
            )
            target_pc = self._apply_preprocess(cropped_target)
            logger.info(
                "Dual-cloud mode: target=%d points, obstacle=%d points",
                target_pc.point_count,
                obstacle_pc.point_count,
            )
            return target_pc, obstacle_pc
        
        target_pc = self._apply_preprocess(raw_pc)
        logger.info("Loaded %d points", target_pc.point_count)
        return target_pc, target_pc
    
    def _get_aabb_transform(self) -> Optional[np.ndarray]:
        """Get AABB transform matrix from config."""
        aabb_transform = getattr(self.config, "aabb_transform", None)
        aabb_center = getattr(self.config, "aabb_center", None)
        aabb_yaw_deg = float(getattr(self.config, "aabb_yaw_deg", 0.0))
        aabb_size = getattr(self.config, "aabb_size", None)
        
        if aabb_size is None:
            return None
        
        if aabb_transform is not None:
            return np.asarray(aabb_transform, dtype=np.float64)
        
        if aabb_center is None:
            raise ValueError("aabb_center is required when aabb_size is set and aabb_transform is not provided")
        
        return xy_rotation_center_to_transform_matrix(aabb_yaw_deg, aabb_center)
    
    def _apply_preprocess(self, pointcloud: PointCloud) -> PointCloud:
        """Apply point cloud denoising and filtering."""
        before_count = pointcloud.point_count
        processed = preprocess_pointcloud(
            pointcloud,
            enable=bool(getattr(self.config, "pointcloud_preprocess_enable", True)),
        )
        if processed.point_count != before_count:
            logger.info(
                "Preprocessing: %d -> %d points",
                before_count,
                processed.point_count,
            )
        return processed


# Import MissionConfig at module level to avoid circular import
from .mission_config import MissionConfig
