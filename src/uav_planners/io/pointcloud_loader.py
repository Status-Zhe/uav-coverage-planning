"""Point cloud loading utilities."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from ..models.pointcloud import PointCloud
from ..utils.pointcloud_aabb import crop_pointcloud_with_transform_box


logger = logging.getLogger(__name__)


class PointCloudLoader:
    """Load point clouds from various file formats.
    
    Supported formats:
    - PCD (Point Cloud Data)
    - PLY (Polygon File Format)
    
    Example:
        loader = PointCloudLoader()
        pc = loader.load("building.pcd")
    """
    
    def load(
        self,
        filepath: Union[str, Path],
        aabb_size: Optional[Tuple[float, float, float]] = None,
        aabb_transform: Optional[np.ndarray] = None,
    ) -> PointCloud:
        """Load point cloud from file.
        
        Args:
            filepath: Path to point cloud file (.pcd or .ply)
            
        Returns:
            Loaded PointCloud object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is unsupported
        """
        filepath = Path(filepath)

        return self._load_pointcloud(
            filepath,
            aabb_size=aabb_size,
            aabb_transform=aabb_transform,
        )

    def _load_pointcloud(
        self,
        filepath: Path,
        aabb_size: Optional[Tuple[float, float, float]] = None,
        aabb_transform: Optional[np.ndarray] = None,
    ) -> PointCloud:
        """Internal load entry with optional AABB crop."""
        
        if not filepath.exists():
            raise FileNotFoundError(f"Point cloud file not found: {filepath}")
        
        suffix = filepath.suffix.lower()
        
        if suffix == '.pcd':
            pointcloud = self._load_pcd(filepath)
        elif suffix == '.ply':
            pointcloud = self._load_ply(filepath)
        else:
            raise ValueError(
                f"Unsupported point cloud format: {suffix}. "
                f"Supported: .pcd, .ply"
            )

        return self.apply_optional_aabb(
            pointcloud,
            aabb_size=aabb_size,
            aabb_transform=aabb_transform,
        )

    def apply_optional_aabb(
        self,
        pointcloud: PointCloud,
        aabb_size: Optional[Tuple[float, float, float]] = None,
        aabb_transform: Optional[np.ndarray] = None,
    ) -> PointCloud:
        """Apply optional AABB/OBB crop to a loaded point cloud."""
        if aabb_size is None:
            return pointcloud

        if aabb_transform is None:
            raise ValueError("aabb_transform is required when aabb_size is provided")

        cropped, local_min, local_max, before_n, after_n = crop_pointcloud_with_transform_box(
            pointcloud,
            transform_matrix=aabb_transform,
            size=aabb_size,
        )
        logger.info(
            "OBB crop applied in loader: %d -> %d points, local_min=%s, local_max=%s",
            before_n,
            after_n,
            local_min.tolist(),
            local_max.tolist(),
        )
        return cropped
    
    def _load_pcd(self, filepath: Path) -> PointCloud:
        """Load PCD format using Open3D.
        
        Args:
            filepath: Path to .pcd file
            
        Returns:
            PointCloud object
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError(
                "open3d is required for PCD loading. "
                "Install: pip install open3d"
            )
        
        pcd = o3d.io.read_point_cloud(str(filepath))
        
        if len(pcd.points) == 0:
            raise ValueError(f"Failed to load point cloud or empty file: {filepath}")
        
        points = np.asarray(pcd.points)
        
        normals = None
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
        
        colors = None
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        
        return PointCloud(
            points=points,
            normals=normals,
            colors=colors,
            source_file=str(filepath)
        )
    
    def _load_ply(self, filepath: Path) -> PointCloud:
        """Load PLY format using Open3D.
        
        Args:
            filepath: Path to .ply file
            
        Returns:
            PointCloud object
        """
        # PLY loading is the same as PCD with Open3D
        return self._load_pcd(filepath)
    
    def save(self, pointcloud: PointCloud, filepath: Union[str, Path]) -> None:
        """Save point cloud to file.
        
        Args:
            pointcloud: Point cloud to save
            filepath: Output file path
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("open3d is required for saving point clouds")
        
        filepath = Path(filepath)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud.points)
        
        if pointcloud.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(pointcloud.normals)
        
        if pointcloud.colors is not None:
            colors_float = pointcloud.colors.astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors_float)
        
        o3d.io.write_point_cloud(str(filepath), pcd)


def load_pointcloud(
    filepath: Union[str, Path],
    aabb_size: Optional[Tuple[float, float, float]] = None,
    aabb_transform: Optional[np.ndarray] = None,
) -> PointCloud:
    """Convenience function to load a point cloud.
    
    Args:
        filepath: Path to point cloud file
        
    Returns:
        Loaded PointCloud
    """
    loader = PointCloudLoader()
    return loader.load(
        filepath,
        aabb_size=aabb_size,
        aabb_transform=aabb_transform,
    )
