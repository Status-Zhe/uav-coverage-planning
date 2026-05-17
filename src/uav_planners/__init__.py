"""UAV Coverage Path Planner library.

A three-stage pipeline for UAV coverage path planning supporting
four algorithms: Boustrophedon, Spiral, Oblique, and Viewpoint.

Example:
    from uav_planners import CoveragePlanner, MissionConfig
    from uav_planners.models import Camera
    
    config = MissionConfig(
        pointcloud_path="building.pcd",
        camera=Camera(...),
        algorithm="boustrophedon"
    )
    
    planner = CoveragePlanner(config)
    result = planner.plan()
"""

__version__ = "0.1.0"

# Core API
from .core.pipeline import CoveragePipeline, CoveragePlanner
from .core.mission_config import MissionConfig
from .core.planner_result import PlannerResult
from .core.data_loader import DataLoader
from .core.config import (
    GeneratorConfig,
    FlightConfig,
    CoverageConfig,
    ViewpointConfig,
    TransitionConfig,
    RegionConfig,
)
from .core.interfaces import Result, ResultStatus

# Models
from .models.camera import Camera
from .models.pointcloud import PointCloud, BoundingBox3D
from .models.waypoint import Waypoint, WaypointAction
from .models.trajectory import Trajectory

# Geometry generators
from .geometry.base_generator import BaseGeometryGenerator, GeneratorConfig
from .geometry.registry import register_generator, get_generator

# Constraints
from .constraints.collision_checker import CollisionChecker
from .constraints.base_validator import ValidationResult

__all__ = [
    # Core
    "CoveragePipeline",
    "CoveragePlanner",
    "MissionConfig",
    "PlannerResult",
    "DataLoader",
    # Configuration (Single Responsibility)
    "GeneratorConfig",
    "FlightConfig",
    "CoverageConfig",
    "ViewpointConfig",
    "TransitionConfig",
    "RegionConfig",
    # Interfaces
    "Result",
    "ResultStatus",
    # Models
    "Camera",
    "PointCloud",
    "BoundingBox3D",
    "Waypoint",
    "WaypointAction",
    "Trajectory",
    # Geometry
    "BaseGeometryGenerator",
    "GeneratorConfig",
    "register_generator",
    "get_generator",
    # Constraints
    "CollisionChecker",
    "ValidationResult",
]
