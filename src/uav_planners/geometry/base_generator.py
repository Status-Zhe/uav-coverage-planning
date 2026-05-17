"""Base class for geometry generators (Stage 1 of pipeline)."""

from abc import ABC, abstractmethod
from typing import List, TypeVar

# Use unified config from core (backward compatible)
from ..core.config import GeneratorConfig

from ..models.waypoint import Waypoint
from ..models.camera import Camera
from ..models.pointcloud import PointCloud

# Re-export for backward compatibility
__all__ = ['BaseGeometryGenerator', 'GeneratorConfig']


class BaseGeometryGenerator(ABC):
    """Abstract base class for geometry generators.
    
    All coverage pattern generators (Boustrophedon, Spiral, Oblique, Viewpoint)
    must inherit from this class and implement the generate() method.
    
    Example:
        @register_generator("my_pattern")
        class MyGenerator(BaseGeometryGenerator):
            @property
            def name(self) -> str:
                return "my_pattern"
            
            def generate(self, pointcloud, camera, config):
                # Generate waypoints
                return waypoints
    """
    
    @abstractmethod
    def generate(
        self,
        pointcloud: PointCloud,
        camera: Camera,
        config: GeneratorConfig
    ) -> List[Waypoint]:
        """Generate raw waypoint sequence for coverage pattern.
        
        Args:
            pointcloud: 3D point cloud of the target structure
            camera: Camera specification for coverage calculation
            config: Generator configuration parameters
            
        Returns:
            List of waypoints representing the coverage path.
            All waypoints should have is_keypoint=True as they are
            the original planned viewpoints (not interpolated).
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this generator.
        
        Returns:
            String identifier used in the registry
        """
        pass
