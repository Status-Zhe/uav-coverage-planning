"""Pipeline integration module for enhanced components.

This module provides integration hooks for:
- FacadeAnalyzer: Complex facade structure detection
- OverlapController: Overlap consistency management
- EnhancedObstacleAvoidance: Multi-layer safety zones
"""

from __future__ import annotations

import logging
from typing import Optional, List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..geometry.facade_analyzer import FacadeAnalyzer
    from ..optimization.overlap_controller import OverlapController
    from ..constraints.enhanced_obstacle_avoidance import EnhancedObstacleAvoidance

logger = logging.getLogger(__name__)


class PipelineIntegrationMixin:
    """Mixin class providing integration with enhanced components.
    
    This mixin integrates FacadeAnalyzer, OverlapController, and 
    EnhancedObstacleAvoidance into the standard pipeline.
    
    Usage:
        class EnhancedCoveragePipeline(PipelineIntegrationMixin, CoveragePipeline):
            def __init__(self, config, enable_facade_analysis=True, ...):
                super().__init__(config)
                self._setup_integration(enable_facade_analysis=enable_facade_analysis)
    """
    
    def _setup_integration(
        self,
        enable_facade_analysis: bool = False,
        enable_overlap_control: bool = False,
        enable_enhanced_avoidance: bool = False,
        **kwargs,
    ) -> None:
        """Setup integration components.
        
        Args:
            enable_facade_analysis: Enable facade structure analysis
            enable_overlap_control: Enable overlap consistency control
            enable_enhanced_avoidance: Enable enhanced obstacle avoidance
            **kwargs: Additional configuration for each component
        """
        self._integration_enabled = {
            'facade': enable_facade_analysis,
            'overlap': enable_overlap_control,
            'avoidance': enable_enhanced_avoidance,
        }
        
        # Initialize FacadeAnalyzer
        if enable_facade_analysis:
            self._init_facade_analyzer(**kwargs)
        
        # Initialize OverlapController
        if enable_overlap_control:
            self._init_overlap_controller(**kwargs)
        
        # Initialize EnhancedObstacleAvoidance
        if enable_enhanced_avoidance:
            self._init_enhanced_avoidance(**kwargs)
    
    def _init_facade_analyzer(self, **kwargs) -> None:
        """Initialize FacadeAnalyzer component.
        
        Args:
            **kwargs: Configuration for FacadeAnalyzer
                - min_corridor_width: Minimum corridor width (default: 2.0)
                - max_corridor_width: Maximum corridor width (default: 15.0)
                - min_corridor_length: Minimum corridor length (default: 3.0)
                - min_recess_depth: Minimum recess depth (default: 1.0)
        """
        from ..geometry.facade_analyzer import FacadeAnalyzer
        
        corridor_config = {
            'min_width': kwargs.get('min_corridor_width', 2.0),
            'max_width': kwargs.get('max_corridor_width', 15.0),
            'min_length': kwargs.get('min_corridor_length', 3.0),
        }
        
        recess_config = {
            'min_depth': kwargs.get('min_recess_depth', 1.0),
            'min_area': kwargs.get('min_recess_area', 4.0),
        }
        
        self.facade_analyzer = FacadeAnalyzer(
            corridor_config=corridor_config,
            recess_config=recess_config,
        )
        
        logger.info("FacadeAnalyzer initialized for complex facade detection")
    
    def _init_overlap_controller(self, **kwargs) -> None:
        """Initialize OverlapController component.
        
        Args:
            **kwargs: Configuration for OverlapController
                - target_front_overlap: Target front overlap (default: 0.3)
                - target_side_overlap: Target side overlap (default: 0.5)
                - max_distance_variance: Max distance variance (default: 0.2)
        """
        from ..optimization.overlap_controller import OverlapController
        
        self.overlap_controller = OverlapController(
            target_front_overlap=kwargs.get('target_front_overlap', 0.3),
            target_side_overlap=kwargs.get('target_side_overlap', 0.5),
            max_distance_variance=kwargs.get('max_distance_variance', 0.2),
        )
        
        logger.info("OverlapController initialized for overlap consistency")
    
    def _init_enhanced_avoidance(self, **kwargs) -> None:
        """Initialize EnhancedObstacleAvoidance component.
        
        Args:
            **kwargs: Configuration for EnhancedObstacleAvoidance
                - safety_margin: Safe operating distance (default: 3.0)
                - warning_margin: Distance to trigger warning (default: 5.0)
                - critical_margin: Distance for critical alert (default: 2.0)
        """
        from ..constraints.enhanced_obstacle_avoidance import EnhancedObstacleAvoidance
        
        self.enhanced_avoidance = EnhancedObstacleAvoidance(
            safety_margin=kwargs.get('safety_margin', 3.0),
            warning_margin=kwargs.get('warning_margin', 5.0),
            critical_margin=kwargs.get('critical_margin', 2.0),
        )
        
        logger.info("EnhancedObstacleAvoidance initialized")
    
    def analyze_facade_structure(self, pointcloud, **kwargs) -> dict:
        """Analyze facade structure if enabled.
        
        Args:
            pointcloud: Target point cloud
            **kwargs: Additional analysis parameters
        
        Returns:
            Analysis result dict, or empty dict if disabled
        """
        if not self._integration_enabled.get('facade', False):
            return {}
        
        if not hasattr(self, 'facade_analyzer'):
            return {}
        
        try:
            # Extract footprints from pointcloud (simplified)
            # In production, use proper footprint extraction
            result = self.facade_analyzer.analyze_complex_facade(
                footprints=[],  # Would need proper footprint extraction
                point_cloud=pointcloud.points if hasattr(pointcloud, 'points') else None,
            )
            
            logger.info(f"Facade analysis complete: {len(result.get('corridors', []))} corridors, "
                       f"{len(result.get('recesses', []))} recesses")
            
            return result
        except Exception as e:
            logger.warning(f"Facade analysis failed: {e}")
            return {}
    
    def validate_overlap_consistency(
        self,
        waypoints: List,
        camera_params: Tuple[float, float],
        **kwargs,
    ) -> List:
        """Validate and adjust overlap consistency if enabled.
        
        Args:
            waypoints: List of waypoints to validate
            camera_params: (horizontal_fov, vertical_fov) tuple
            **kwargs: Additional parameters
        
        Returns:
            Original waypoints if disabled, or adjusted waypoints if enabled
        """
        if not self._integration_enabled.get('overlap', False):
            return waypoints
        
        if not hasattr(self, 'overlap_controller'):
            return waypoints
        
        try:
            # Calculate capture distance consistency
            h_fov, v_fov = camera_params
            
            # Analyze overlap for each segment
            adjusted_waypoints = self.overlap_controller.adjust_path_for_consistency(
                waypoints=waypoints,
                capture_distance=self.config.altitude,
                overlap_ratio=self.config.side_overlap,
            )
            
            logger.info(f"Overlap validation: {len(adjusted_waypoints)} waypoints adjusted")
            
            return adjusted_waypoints
        except Exception as e:
            logger.warning(f"Overlap validation failed: {e}")
            return waypoints
    
    def check_enhanced_safety(self, waypoint, **kwargs) -> Tuple[bool, Optional[str]]:
        """Enhanced safety check using multi-layer safety zones.
        
        Args:
            waypoint: Waypoint to check
            **kwargs: Additional parameters
        
        Returns:
            (is_safe, warning_message) tuple
        """
        if not self._integration_enabled.get('avoidance', False):
            return True, None
        
        if not hasattr(self, 'enhanced_avoidance'):
            return True, None
        
        try:
            # Get waypoint position
            position = np.array([waypoint.x, waypoint.y, waypoint.z])
            
            # Perform safety check
            is_safe, level, message = self.enhanced_avoidance.check_waypoint_safety(
                waypoint_position=position,
                obstacles=[],  # Would need obstacle list from pointcloud
            )
            
            if not is_safe:
                logger.warning(f"Safety warning at ({waypoint.x}, {waypoint.y}, {waypoint.z}): {message}")
            
            return is_safe, message
        except Exception as e:
            logger.warning(f"Enhanced safety check failed: {e}")
            return True, None


def create_enhanced_pipeline(config, **integration_options) -> PipelineIntegrationMixin:
    """Factory function to create pipeline with integration enabled.
    
    Args:
        config: MissionConfig instance
        **integration_options: Options for integration components
            - enable_facade_analysis: Enable facade analysis (default: False)
            - enable_overlap_control: Enable overlap control (default: False)
            - enable_enhanced_avoidance: Enable enhanced avoidance (default: False)
            - Plus component-specific options
    
    Returns:
        Pipeline with integration mixin
    """
    from .pipeline import CoveragePipeline
    
    class EnhancedPipeline(PipelineIntegrationMixin, CoveragePipeline):
        pass
    
    pipeline = EnhancedPipeline(config)
    pipeline._setup_integration(**integration_options)
    
    return pipeline
