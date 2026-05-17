"""Point cloud model for UAV path planning."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class BoundingBox3D:
    """Axis-aligned 3D bounding box."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    
    @property
    def width(self) -> float:
        return self.max_x - self.min_x
    
    @property
    def height(self) -> float:
        return self.max_y - self.min_y
    
    @property
    def depth(self) -> float:
        return self.max_z - self.min_z
    
    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth
    
    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2
        )


@dataclass
class PointCloud:
    """3D point cloud representation with optional normals and colors.
    
    Attributes:
        points: XYZ coordinates, shape (N, 3) in meters
        normals: Normal vectors, shape (N, 3), optional
        colors: RGB values [0-255], shape (N, 3), optional
        bounds: Axis-aligned bounding box (auto-calculated)
        density: Points per cubic meter (auto-calculated)
        source_file: Original file path, optional
    """
    points: np.ndarray
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    source_file: Optional[str] = None
    coordinate_frame: str = "world"
    enu_origin_ecef: Optional[Tuple[float, float, float]] = None
    bounds: BoundingBox3D = field(init=False)
    
    def __post_init__(self):
        """Calculate bounds after initialization."""
        self.bounds = BoundingBox3D(
            min_x=float(np.min(self.points[:, 0])),
            max_x=float(np.max(self.points[:, 0])),
            min_y=float(np.min(self.points[:, 1])),
            max_y=float(np.max(self.points[:, 1])),
            min_z=float(np.min(self.points[:, 2])),
            max_z=float(np.max(self.points[:, 2])),
        )
    
    @property
    def point_count(self) -> int:
        """Number of points in the cloud."""
        return len(self.points)
    
    @property
    def density(self) -> float:
        """Points per cubic meter."""
        volume = self.bounds.volume
        if volume > 0:
            return self.point_count / volume
        return 0.0
    
    def query_nearest(
        self,
        query_points: np.ndarray,
        k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for query points.
        
        Args:
            query_points: Points to query, shape (M, 3)
            k: Number of nearest neighbors to find
            
        Returns:
            distances: Distances to nearest neighbors, shape (M, k)
            indices: Indices of nearest neighbors, shape (M, k)
        """
        from scipy.spatial import cKDTree
        
        tree = cKDTree(self.points)
        distances, indices = tree.query(query_points, k=k)
        return distances, indices
