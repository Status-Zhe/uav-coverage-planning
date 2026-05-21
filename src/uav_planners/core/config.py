"""Configuration classes with single responsibility principle.

This module provides focused configuration classes for different
aspects of the coverage planning system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
from enum import Enum


class OverlapMode(Enum):
    """Overlap calculation mode."""
    SIDE = "side"
    FRONT = "front"
    BOTH = "both"


class LayerOrder(Enum):
    """Layer traversal order."""
    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"
    OUTSIDE_IN = "outside_in"
    INSIDE_OUT = "inside_out"


class HullMethod(Enum):
    """Hull generation method."""
    ALPHA = "alpha"
    CONVEX = "convex"


@dataclass
class FlightConfig:
    """Flight operation parameters.
    
    Focus: Core flight safety and performance.
    """
    altitude: float = 50.0
    """Flight altitude above ground in meters."""
    
    speed_ms: float = 5.0
    """Target flight speed in meters per second."""
    
    safety_distance: float = 3.0
    """Minimum distance to obstacles in meters."""
    
    min_flight_altitude_m: float = 5.0
    """Minimum safe flight altitude above ground."""


@dataclass
class CoverageConfig:
    """Coverage quality parameters.
    
    Focus: Image overlap and coverage ratio control.
    """
    side_overlap: float = 0.7
    """Side overlap ratio (0-1), default 70%."""
    
    front_overlap: float = 0.8
    """Front overlap ratio (0-1), default 80%."""
    
    coverage_threshold: float = 0.95
    """Required coverage ratio for viewpoint planning (0-1)."""
    
    oblique_dst_srf: float = 5.0
    """Distance between parallel lines in oblique pattern (meters)."""
    
    oblique_min_altitude: float = 5.0
    """Minimum altitude for oblique photography (meters)."""


@dataclass
class ViewpointConfig:
    """Viewpoint generation parameters.
    
    Focus: Layer-based viewpoint planning for complex structures.
    """
    # Layer management
    viewpoint_layer_height_step_m: float = 2.5
    """Height step between layers (meters)."""
    
    viewpoint_layer_order: LayerOrder = LayerOrder.BOTTOM_UP
    """Order of traversing layers."""
    
    viewpoint_layer_area_jump_ratio: float = 1.6
    """Area ratio threshold for new layer detection."""
    
    viewpoint_layer_insert_max_global: int = 8
    """Maximum viewpoints to insert globally per layer."""
    
    # Boundary settings
    viewpoint_boundary_expand_m: float = 4.0
    """Distance to expand boundary outward (meters)."""
    
    viewpoint_min_altitude: float = 0.0
    """Minimum altitude for viewpoints."""
    
    viewpoint_beyond_altitude: float = 0.0
    """Altitude above top of structure for final pass."""
    
    # Ring/arc settings
    viewpoint_ring_arc_step_m: float = 2.5
    """Arc step for ring viewpoints (meters)."""
    
    viewpoint_min_points_per_layer: int = 8
    """Minimum ring points per layer."""
    
    # Shape generation
    viewpoint_shape_method: HullMethod = HullMethod.ALPHA
    """Method for generating footprint hull."""
    
    viewpoint_alpha: float = 6.0
    """Alpha value for alpha-shape hull generation."""
    
    viewpoint_shape_use_full_points: bool = False
    """Use all points for shape generation."""
    
    viewpoint_shape_roundness_m: float = 0.0
    """Roundness parameter for shape smoothing."""
    
    # Hull settings
    viewpoint_hull_use_full_points: bool = False
    """Use all points for hull generation."""
    
    viewpoint_hull_roundness_m: float = 0.0
    """Roundness for hull smoothing."""
    
    # Height slicing
    viewpoint_use_height_slice: bool = True
    """Enable height slicing for complex structures."""
    
    viewpoint_slice_thickness_pct: float = 0.2
    """Slice thickness as percentage of layer height."""
    
    viewpoint_overlap_pct: float = 0.2
    """Overlap between adjacent slices (percentage)."""


@dataclass
class TransitionConfig:
    """Transition path generation parameters.
    
    Focus: Connecting viewpoints with safe paths.
    """
    start_waypoint: Optional[Any] = None
    """Starting waypoint for single transition."""
    
    goal_waypoint: Optional[Any] = None
    """Goal waypoint for single transition."""
    
    transitions: List[Tuple[Any, Any]] = field(default_factory=list)
    """List of (start, goal) waypoint pairs for batch processing."""
    
    # Grid search
    transition_grid_resolution: float = 2.0
    """Grid resolution for graph search (meters)."""
    
    transition_max_expansions: int = 30000
    """Maximum graph expansions per search."""
    
    transition_goal_tolerance: float = 2.0
    """Goal reach tolerance (meters)."""
    
    # Waypoint output
    transition_waypoint_spacing: float = 4.0
    """Spacing for densified output points (meters)."""
    
    # Lateral optimization
    transition_prefer_lateral_before_altitude: bool = True
    """Prefer lateral movement before altitude changes."""
    
    transition_lateral_offset_min_m: float = 6.0
    """Minimum lateral offset from direct path (meters)."""
    
    transition_lateral_offset_max_m: float = 30.0
    """Maximum lateral offset (meters)."""
    
    transition_lateral_offset_step_m: float = 4.0
    """Step size for lateral offset search (meters)."""
    
    transition_lateral_max_candidates: int = 16
    """Maximum lateral candidates to evaluate."""
    
    transition_lateral_turn_penalty_weight: float = 0.15
    """Weight for turn penalty in lateral optimization."""
    
    # Fallback
    transition_enable_theta_star_fallback: bool = False
    """Enable Theta* fallback if A* fails."""


@dataclass
class RegionConfig:
    """Region-only mode configuration.
    
    Focus: Simple rectangular coverage without point cloud.
    """
    enabled: bool = False
    """Enable region-only mode."""
    
    area_rect_xy: Tuple[float, float, float, float] = (0, 0, 100, 100)
    """Rectangle in XY plane: (xmin, ymin, xmax, ymax)."""
    
    ground_z: float = 0.0
    """Ground plane Z coordinate."""
    
    # Distance settings
    global_distance_m: Optional[float] = None
    """Global capture distance override."""


@dataclass
class SpiralConfig:
    """Spiral circle-only parameters.

    When provided, the spiral generator uses these values instead of
    fitting a cylinder from a point cloud.
    """
    spiral_center_xy: Optional[Tuple[float, float]] = None
    """Circle center in XY plane."""

    spiral_radius: Optional[float] = None
    """Circle radius in meters."""

    spiral_start_z: Optional[float] = None
    """Spiral start elevation (Z)."""

    spiral_height: Optional[float] = None
    """Total spiral height in meters."""


@dataclass
class CylinderConfig:
    """Cylinder path parameters.

    When provided, the cylinder generator uses these values instead of
    fitting a cylinder from a point cloud.
    """
    cylinder_center_xy: Optional[Tuple[float, float]] = None
    """Cylinder center in XY plane."""

    cylinder_radius: Optional[float] = None
    """Cylinder radius in meters."""

    cylinder_start_z: Optional[float] = None
    """Cylinder start elevation (Z)."""

    cylinder_height: Optional[float] = None
    """Cylinder height in meters."""

    cylinder_mode: str = "horizontal"
    """Path mode: horizontal (rings) or vertical (boustrophedon)."""

    cylinder_ring_spacing_m: Optional[float] = None
    """Optional ring spacing in meters (horizontal mode)."""

    cylinder_ring_count: Optional[int] = None
    """Optional ring count (horizontal mode)."""

    cylinder_strip_spacing_m: Optional[float] = None
    """Optional arc-length spacing between strips in meters (vertical mode)."""

    cylinder_strip_count: Optional[int] = None
    """Optional strip count (vertical mode)."""

    cylinder_angle_start_deg: float = 0.0
    """Start angle for rings/strips in degrees."""


@dataclass
class OnePlaneConfig:
    """Oblique one-plane input parameters.

    Provides a polygon in 3D that defines a single plane for scanning.
    """
    oneplane_polygon_xyz: Optional[List[Tuple[float, float, float]]] = None
    """Polygon vertices in 3D (must be coplanar)."""

    oneplane_plane_tolerance: float = 0.02
    """Max distance to plane for coplanarity check (meters)."""

    oneplane_face_normal_sign: float = -1.0
    """Normal sign for camera facing direction (-1 to face opposite normal)."""

    oneplane_heading_yaw_offset_deg: float = 0.0
    """Yaw offset around normal (degrees)."""


# Import for backward compatibility and type hints
from typing import Union

# Type alias for unified config
ConfigType = Union[FlightConfig, CoverageConfig, ViewpointConfig, TransitionConfig, RegionConfig]


class GeneratorConfig:
    """Unified configuration for geometry generators.
    
    This is a composite configuration that holds all sub-configurations.
    Use the specialized configs for focused control, or this unified
    config for convenience.
    
    Example:
        # Full configuration
        config = GeneratorConfig(
            flight=FlightConfig(altitude=60.0, speed_ms=8.0),
            coverage=CoverageConfig(side_overlap=0.6),
            viewpoint=ViewpointConfig(viewpoint_alpha=8.0),
            transition=TransitionConfig(),
        )
        
        # Quick setup (uses defaults)
        config = GeneratorConfig()
        
        # Backward compatible: keyword arguments
        config = GeneratorConfig(altitude=50.0, side_overlap=0.7)
        
        # Access sub-configs
        alt = config.flight.altitude
        config.flight.altitude = 70.0
    """
    # Sub-configurations
    flight: FlightConfig = field(default_factory=FlightConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    viewpoint: ViewpointConfig = field(default_factory=ViewpointConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    spiral: SpiralConfig = field(default_factory=SpiralConfig)
    cylinder: CylinderConfig = field(default_factory=CylinderConfig)
    oneplane: OnePlaneConfig = field(default_factory=OnePlaneConfig)
    
    # Legacy attributes for backward compatibility
    # These are auto-mapped to sub-configs via properties
    _legacy_attrs: List[str] = field(default_factory=lambda: [
        'altitude', 'speed_ms', 'safety_distance', 'min_flight_altitude_m',
        'side_overlap', 'front_overlap', 'coverage_threshold', 'oblique_dst_srf', 'oblique_min_altitude',
        'viewpoint_layer_height_step_m', 'viewpoint_layer_order', 'viewpoint_boundary_expand_m',
        'viewpoint_ring_arc_step_m', 'viewpoint_min_points_per_layer', 'viewpoint_layer_area_jump_ratio',
        'viewpoint_layer_insert_max_global', 'viewpoint_min_altitude', 'viewpoint_beyond_altitude',
        'viewpoint_shape_use_full_points', 'viewpoint_shape_roundness_m', 'viewpoint_shape_method',
        'viewpoint_hull_use_full_points', 'viewpoint_hull_roundness_m', 'viewpoint_alpha',
        'viewpoint_use_height_slice', 'viewpoint_slice_thickness_pct', 'viewpoint_overlap_pct',
        'start_waypoint', 'goal_waypoint', 'transitions', 'transition_grid_resolution',
        'transition_max_expansions', 'transition_goal_tolerance', 'transition_waypoint_spacing',
        'transition_prefer_lateral_before_altitude', 'transition_lateral_offset_min_m',
        'transition_lateral_offset_max_m', 'transition_lateral_offset_step_m',
        'transition_lateral_max_candidates', 'transition_lateral_turn_penalty_weight',
        'transition_enable_theta_star_fallback',
        'region_only_enabled', 'coverage_area_rect_xy', 'region_ground_z',
        'global_distance_m',
        'spiral_center_xy', 'spiral_radius', 'spiral_start_z', 'spiral_height',
        'cylinder_center_xy', 'cylinder_radius', 'cylinder_start_z', 'cylinder_height',
        'cylinder_mode', 'cylinder_ring_spacing_m', 'cylinder_ring_count',
        'cylinder_strip_spacing_m', 'cylinder_strip_count', 'cylinder_angle_start_deg',
        'oneplane_polygon_xyz', 'oneplane_plane_tolerance',
        'oneplane_face_normal_sign', 'oneplane_heading_yaw_offset_deg',
    ], repr=False)
    
    # Custom init to support backward-compatible keyword arguments
    def __init__(
        self,
        flight: Any = None,
        coverage: Any = None,
        viewpoint: Any = None,
        transition: Any = None,
        region: Any = None,
        spiral: Any = None,
        cylinder: Any = None,
        oneplane: Any = None,
        **kwargs,
    ):
        """Initialize with sub-configs or legacy keyword arguments.
        
        Supports both:
        - Sub-config initialization: GeneratorConfig(flight=FlightConfig(...))
        - Sub-config from dict: GeneratorConfig(flight={'altitude': 100.0})
        - Legacy keyword args: GeneratorConfig(altitude=50.0, side_overlap=0.7)
        """
        # Handle sub-configs (can be FlightConfig instance or dict)
        if flight is None:
            self.flight = FlightConfig()
        elif isinstance(flight, dict):
            self.flight = FlightConfig(**flight)
        else:
            self.flight = flight
        
        if coverage is None:
            self.coverage = CoverageConfig()
        elif isinstance(coverage, dict):
            self.coverage = CoverageConfig(**coverage)
        else:
            self.coverage = coverage
        
        if viewpoint is None:
            self.viewpoint = ViewpointConfig()
        elif isinstance(viewpoint, dict):
            self.viewpoint = ViewpointConfig(**viewpoint)
        else:
            self.viewpoint = viewpoint
        
        if transition is None:
            self.transition = TransitionConfig()
        elif isinstance(transition, dict):
            self.transition = TransitionConfig(**transition)
        else:
            self.transition = transition
        
        if region is None:
            self.region = RegionConfig()
        elif isinstance(region, dict):
            self.region = RegionConfig(**region)
        else:
            self.region = region

        if spiral is None:
            self.spiral = SpiralConfig()
        elif isinstance(spiral, dict):
            self.spiral = SpiralConfig(**spiral)
        else:
            self.spiral = spiral

        if cylinder is None:
            self.cylinder = CylinderConfig()
        elif isinstance(cylinder, dict):
            self.cylinder = CylinderConfig(**cylinder)
        else:
            self.cylinder = cylinder

        if oneplane is None:
            self.oneplane = OnePlaneConfig()
        elif isinstance(oneplane, dict):
            self.oneplane = OnePlaneConfig(**oneplane)
        else:
            self.oneplane = oneplane
        
        # Handle legacy keyword arguments
        if kwargs:
            # Build a flat dict of all legacy attrs
            legacy_map = self._build_legacy_map()
            for key, value in kwargs.items():
                if key in legacy_map:
                    setattr(self, key, value)
    
    def _build_legacy_map(self) -> dict:
        """Build mapping of legacy attrs to their sub-configs."""
        flight_attrs = {'altitude', 'speed_ms', 'safety_distance', 'min_flight_altitude_m'}
        coverage_attrs = {'side_overlap', 'front_overlap', 'coverage_threshold', 'oblique_dst_srf', 'oblique_min_altitude'}
        viewpoint_attrs = {
            'viewpoint_layer_height_step_m', 'viewpoint_layer_order', 'viewpoint_boundary_expand_m',
            'viewpoint_ring_arc_step_m', 'viewpoint_min_points_per_layer', 'viewpoint_layer_area_jump_ratio',
            'viewpoint_layer_insert_max_global', 'viewpoint_min_altitude', 'viewpoint_beyond_altitude',
            'viewpoint_shape_use_full_points', 'viewpoint_shape_roundness_m', 'viewpoint_shape_method',
            'viewpoint_hull_use_full_points', 'viewpoint_hull_roundness_m', 'viewpoint_alpha',
            'viewpoint_use_height_slice', 'viewpoint_slice_thickness_pct', 'viewpoint_overlap_pct',
        }
        transition_attrs = {
            'start_waypoint', 'goal_waypoint', 'transitions', 'transition_grid_resolution',
            'transition_max_expansions', 'transition_goal_tolerance', 'transition_waypoint_spacing',
            'transition_prefer_lateral_before_altitude', 'transition_lateral_offset_min_m',
            'transition_lateral_offset_max_m', 'transition_lateral_offset_step_m',
            'transition_lateral_max_candidates', 'transition_lateral_turn_penalty_weight',
            'transition_enable_theta_star_fallback',
        }
        region_attrs = {'region_only_enabled', 'coverage_area_rect_xy', 'region_ground_z', 'global_distance_m'}
        spiral_attrs = {'spiral_center_xy', 'spiral_radius', 'spiral_start_z', 'spiral_height'}
        cylinder_attrs = {
            'cylinder_center_xy', 'cylinder_radius', 'cylinder_start_z', 'cylinder_height',
            'cylinder_mode', 'cylinder_ring_spacing_m', 'cylinder_ring_count',
            'cylinder_strip_spacing_m', 'cylinder_strip_count', 'cylinder_angle_start_deg',
        }
        oneplane_attrs = {
            'oneplane_polygon_xyz', 'oneplane_plane_tolerance',
            'oneplane_face_normal_sign', 'oneplane_heading_yaw_offset_deg',
        }
        
        return {
            *flight_attrs,
            *coverage_attrs,
            *viewpoint_attrs,
            *transition_attrs,
            *region_attrs,
            *spiral_attrs,
            *cylinder_attrs,
            *oneplane_attrs,
        }
    
    def __getattr__(self, name: str) -> Any:
        """Provide backward-compatible access to legacy attributes."""
        # Skip internal attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        
        # Map legacy attributes to sub-configs
        flight_attrs = {'altitude', 'speed_ms', 'safety_distance', 'min_flight_altitude_m'}
        coverage_attrs = {'side_overlap', 'front_overlap', 'coverage_threshold', 'oblique_dst_srf', 'oblique_min_altitude'}
        viewpoint_attrs = {
            'viewpoint_layer_height_step_m', 'viewpoint_layer_order', 'viewpoint_boundary_expand_m',
            'viewpoint_ring_arc_step_m', 'viewpoint_min_points_per_layer', 'viewpoint_layer_area_jump_ratio',
            'viewpoint_layer_insert_max_global', 'viewpoint_min_altitude', 'viewpoint_beyond_altitude',
            'viewpoint_shape_use_full_points', 'viewpoint_shape_roundness_m', 'viewpoint_shape_method',
            'viewpoint_hull_use_full_points', 'viewpoint_hull_roundness_m', 'viewpoint_alpha',
            'viewpoint_use_height_slice', 'viewpoint_slice_thickness_pct', 'viewpoint_overlap_pct',
        }
        transition_attrs = {
            'start_waypoint', 'goal_waypoint', 'transitions', 'transition_grid_resolution',
            'transition_max_expansions', 'transition_goal_tolerance', 'transition_waypoint_spacing',
            'transition_prefer_lateral_before_altitude', 'transition_lateral_offset_min_m',
            'transition_lateral_offset_max_m', 'transition_lateral_offset_step_m',
            'transition_lateral_max_candidates', 'transition_lateral_turn_penalty_weight',
            'transition_enable_theta_star_fallback',
        }
        region_attrs = {'region_only_enabled', 'coverage_area_rect_xy', 'region_ground_z', 'global_distance_m'}
        spiral_attrs = {'spiral_center_xy', 'spiral_radius', 'spiral_start_z', 'spiral_height'}
        cylinder_attrs = {
            'cylinder_center_xy', 'cylinder_radius', 'cylinder_start_z', 'cylinder_height',
            'cylinder_mode', 'cylinder_ring_spacing_m', 'cylinder_ring_count',
            'cylinder_strip_spacing_m', 'cylinder_strip_count', 'cylinder_angle_start_deg',
        }
        oneplane_attrs = {
            'oneplane_polygon_xyz', 'oneplane_plane_tolerance',
            'oneplane_face_normal_sign', 'oneplane_heading_yaw_offset_deg',
        }
        
        if name in flight_attrs:
            return getattr(self.flight, name)
        elif name in coverage_attrs:
            return getattr(self.coverage, name)
        elif name in viewpoint_attrs:
            return getattr(self.viewpoint, name)
        elif name in transition_attrs:
            return getattr(self.transition, name)
        elif name in region_attrs:
            return getattr(self.region, name)
        elif name in spiral_attrs:
            return getattr(self.spiral, name)
        elif name in cylinder_attrs:
            return getattr(self.cylinder, name)
        elif name in oneplane_attrs:
            return getattr(self.oneplane, name)
        
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Provide backward-compatible setting of legacy attributes."""
        # Skip _legacy_attrs for normal attribute setting
        if name == '_legacy_attrs':
            super().__setattr__(name, value)
            return
        
        flight_attrs = {'altitude', 'speed_ms', 'safety_distance', 'min_flight_altitude_m'}
        coverage_attrs = {'side_overlap', 'front_overlap', 'coverage_threshold', 'oblique_dst_srf', 'oblique_min_altitude'}
        viewpoint_attrs = {
            'viewpoint_layer_height_step_m', 'viewpoint_layer_order', 'viewpoint_boundary_expand_m',
            'viewpoint_ring_arc_step_m', 'viewpoint_min_points_per_layer', 'viewpoint_layer_area_jump_ratio',
            'viewpoint_layer_insert_max_global', 'viewpoint_min_altitude', 'viewpoint_beyond_altitude',
            'viewpoint_shape_use_full_points', 'viewpoint_shape_roundness_m', 'viewpoint_shape_method',
            'viewpoint_hull_use_full_points', 'viewpoint_hull_roundness_m', 'viewpoint_alpha',
            'viewpoint_use_height_slice', 'viewpoint_slice_thickness_pct', 'viewpoint_overlap_pct',
        }
        transition_attrs = {
            'start_waypoint', 'goal_waypoint', 'transitions', 'transition_grid_resolution',
            'transition_max_expansions', 'transition_goal_tolerance', 'transition_waypoint_spacing',
            'transition_prefer_lateral_before_altitude', 'transition_lateral_offset_min_m',
            'transition_lateral_offset_max_m', 'transition_lateral_offset_step_m',
            'transition_lateral_max_candidates', 'transition_lateral_turn_penalty_weight',
            'transition_enable_theta_star_fallback',
        }
        region_attrs = {'region_only_enabled', 'coverage_area_rect_xy', 'region_ground_z', 'global_distance_m'}
        spiral_attrs = {'spiral_center_xy', 'spiral_radius', 'spiral_start_z', 'spiral_height'}
        cylinder_attrs = {
            'cylinder_center_xy', 'cylinder_radius', 'cylinder_start_z', 'cylinder_height',
            'cylinder_mode', 'cylinder_ring_spacing_m', 'cylinder_ring_count',
            'cylinder_strip_spacing_m', 'cylinder_strip_count', 'cylinder_angle_start_deg',
        }
        oneplane_attrs = {
            'oneplane_polygon_xyz', 'oneplane_plane_tolerance',
            'oneplane_face_normal_sign', 'oneplane_heading_yaw_offset_deg',
        }
        
        if name in flight_attrs:
            object.__setattr__(self.flight, name, value)
        elif name in coverage_attrs:
            object.__setattr__(self.coverage, name, value)
        elif name in viewpoint_attrs:
            object.__setattr__(self.viewpoint, name, value)
        elif name in transition_attrs:
            object.__setattr__(self.transition, name, value)
        elif name in region_attrs:
            object.__setattr__(self.region, name, value)
        elif name in spiral_attrs:
            object.__setattr__(self.spiral, name, value)
        elif name in cylinder_attrs:
            object.__setattr__(self.cylinder, name, value)
        elif name in oneplane_attrs:
            object.__setattr__(self.oneplane, name, value)
        else:
            super().__setattr__(name, value)
