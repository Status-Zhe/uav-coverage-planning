"""Spiral trajectory generator for cylindrical structures."""

import math
from typing import List, Tuple
import numpy as np

from ..base_generator import BaseGeometryGenerator, GeneratorConfig
from ..registry import register_generator
from ...models.waypoint import Waypoint, WaypointAction
from ...models.camera import Camera
from ...models.pointcloud import PointCloud


@register_generator("spiral")
class SpiralGenerator(BaseGeometryGenerator):
    """Spiral coverage pattern generator for cylindrical structures.
    
    Generates a helical scanning pattern suitable for:
    - Chimneys and cooling towers
    - Industrial tanks and silos
    - Telecommunication towers
    - Any cylindrical or conical structure
    
    Algorithm:
    1. Fit cylinder to point cloud to find center and radius
    2. Calculate spiral parameters (pitch, turns, height)
    3. Generate helical path using parametric equations
    4. Camera always points toward cylinder center axis
    
    Mathematical model:
    - Cylindrical spiral: r = constant
    - Conical spiral: r varies linearly with height
    - x = cx + r * cos(θ)
    - y = cy + r * sin(θ)
    - z = cz + (pitch * θ) / (2π)
    
    Reference:
        Spiral trajectories for building inspection
    """
    
    @property
    def name(self) -> str:
        return "spiral"
    
    def generate(
        self,
        pointcloud: PointCloud,
        camera: Camera,
        config: GeneratorConfig
    ) -> List[Waypoint]:
        """Generate spiral scanning pattern.
        
        Args:
            pointcloud: Cylindrical structure point cloud
            camera: Camera specification
            config: Generator configuration
            
        Returns:
            Spiral waypoints
        """
        spiral_center_xy = getattr(config, "spiral_center_xy", None)
        spiral_radius = getattr(config, "spiral_radius", None)
        spiral_start_z = getattr(config, "spiral_start_z", None)
        spiral_height = getattr(config, "spiral_height", None)

        if (
            spiral_center_xy is not None
            and spiral_radius is not None
            and spiral_start_z is not None
            and spiral_height is not None
        ):
            center_xy = (float(spiral_center_xy[0]), float(spiral_center_xy[1]))
            base_radius = float(spiral_radius)
            height = float(spiral_height)
            z_min = None
            z_start_override = float(spiral_start_z)
            gimbal_pitch_override = 0.0
        else:
            # Step 1: Fit cylinder to point cloud
            center_xy, base_radius, height = self._fit_cylinder(pointcloud)
            z_min = float(np.min(pointcloud.points[:, 2]))
            z_start_override = None
            gimbal_pitch_override = None
        
        # Step 2: Calculate spiral parameters
        spiral_config = self._compute_spiral_params(
            base_radius,
            height,
            camera,
            config,
            z_min=z_min,
            z_start_override=z_start_override,
        )
        spiral_config["gimbal_pitch_deg"] = gimbal_pitch_override
        
        # Step 3: Generate spiral path
        waypoints = self._generate_helical_path(
            center_xy, spiral_config, config
        )
        
        return waypoints
    
    def _fit_cylinder(self, pointcloud: PointCloud) -> Tuple[Tuple[float, float], float, float]:
        """Fit cylinder to point cloud using bounding box approach.
        
        Improved method:
        1. Compute 3D bounding box of point cloud
        2. Use bounding box center as cylinder center
        3. Estimate radius from bounding box dimensions
        4. Compute height from bounding box Z range
        
        Args:
            pointcloud: Input point cloud
            
        Returns:
            Tuple of (center_xy, radius, height)
        """
        points = pointcloud.points
        
        # Method 1: Simple bounding box (axis-aligned)
        min_xyz = np.min(points, axis=0)
        max_xyz = np.max(points, axis=0)
        
        # Bounding box center
        center_x = (min_xyz[0] + max_xyz[0]) / 2
        center_y = (min_xyz[1] + max_xyz[1]) / 2
        center_z = (min_xyz[2] + max_xyz[2]) / 2
        
        # Bounding box dimensions
        width_x = max_xyz[0] - min_xyz[0]
        width_y = max_xyz[1] - min_xyz[1]
        height = max_xyz[2] - min_xyz[2]
        
        # Estimate radius from bounding box (assuming circular cross-section)
        # Use the average of width_x and width_y, then half for radius
        diameter = (width_x + width_y) / 2
        radius = diameter / 2.5
        
        # Alternative: Use the maximum dimension (more aggressive)
        # radius = max(width_x, width_y) / 2
        
        return (float(center_x), float(center_y)), float(radius), float(height)
    
    def _compute_spiral_params(
        self,
        base_radius: float,
        height: float,
        camera: Camera,
        config: GeneratorConfig,
        z_min: float = 0.0,
        z_start_override: float = None,
    ) -> dict:
        """Compute spiral parameters.
        
        Args:
            base_radius: Cylinder base radius
            height: Structure height
            camera: Camera specification
            config: Generator configuration
            
        Returns:
            Dictionary of spiral parameters
        """
        # Flight radius: base radius + stand-off distance
        global_distance = getattr(config, "global_distance_m", None)
        if global_distance is not None:
            flight_radius = base_radius + float(global_distance)
        else:
            print(f"Warning: global_distance_m not set, using default expansion for flight radius." )
            flight_radius = base_radius * 1.8
        
        # Calculate pitch based on camera FOV for continuous coverage
        fov_v_rad = math.radians(camera.fov_vertical_deg)
        
        # At flight distance, compute ground coverage per rotation
        altitude_factor = config.altitude / max(base_radius, 1.0)
        vertical_coverage = 2 * flight_radius * math.tan(fov_v_rad / 2)
        
        # Pitch for 70% overlap
        pitch = vertical_coverage * (1 - config.front_overlap)
        pitch = max(pitch, 2.0)  # Minimum 2m pitch
        
        # Calculate number of turns
        turns = height / pitch
        
        # Total angle
        total_angle = turns * 2 * math.pi
        
        # Number of waypoints (at least 20 per rotation)
        num_points = max(int(turns * 20), int(total_angle * 5))
        
        min_alt_offset = getattr(config, "min_flight_altitude_m", None)
        if min_alt_offset is None:
            min_alt_offset = config.altitude * 0.5

        if z_start_override is not None:
            z_start = float(z_start_override)
        else:
            min_alt_offset = getattr(config, "min_flight_altitude_m", None)
            if min_alt_offset is None:
                min_alt_offset = config.altitude * 0.5
            z_start = z_min + float(min_alt_offset)

        return {
            "center": None,  # Will be set later
            "flight_radius": flight_radius,
            "base_radius": base_radius,
            "height": height,
            "pitch": pitch,
            "turns": turns,
            "total_angle": total_angle,
            "num_points": num_points,
            "z_start": z_start,
        }
    
    def _generate_helical_path(
        self,
        center_xy: Tuple[float, float],
        spiral_config: dict,
        config: GeneratorConfig
    ) -> List[Waypoint]:
        """Generate helical waypoint path.
        
        Args:
            center_xy: Cylinder center (x, y)
            spiral_config: Spiral parameters
            config: Generator configuration
            
        Returns:
            List of spiral waypoints
        """
        cx, cy = center_xy
        radius = spiral_config["flight_radius"]
        pitch = spiral_config["pitch"]
        num_points = spiral_config["num_points"]
        total_angle = spiral_config["total_angle"]
        z_start = spiral_config["z_start"]
        
        waypoints = []
        prev_heading = None
        
        # Generate points along helix
        for i in range(num_points):
            t = i / (num_points - 1)  # 0 to 1
            theta = t * total_angle
            
            # Helix parametric equations
            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)
            z = z_start + (pitch * theta) / (2 * math.pi)
            
            # Camera heading: point toward cylinder center
            # atan2(y, x) gives angle from x-axis, but heading is from y-axis (North)
            # Point from current position toward center
            dx = cx - x
            dy = cy - y
            heading = self._unwrap_heading_to_previous(
                math.degrees(math.atan2(dx, dy)),
                prev_heading,
            )
            prev_heading = heading
            
            gimbal_pitch = spiral_config.get("gimbal_pitch_deg")
            if gimbal_pitch is None:
                gimbal_pitch = -45.0
            
            wp = Waypoint(
                x=float(x),
                y=float(y),
                z=float(z),
                heading_deg=heading,
                gimbal_pitch_deg=gimbal_pitch,
                speed_ms=config.speed_ms,
                action=WaypointAction.SHOOT,
                dwell_time_s=0.0,
                is_keypoint=True
            )
            waypoints.append(wp)
        
        return waypoints
    
    def generate_conical_spiral(
        self,
        pointcloud: PointCloud,
        camera: Camera,
        config: GeneratorConfig,
        top_radius_ratio: float = 0.5
    ) -> List[Waypoint]:
        """Generate conical spiral (for tapered structures).
        
        Args:
            pointcloud: Conical structure point cloud
            camera: Camera specification
            config: Generator configuration
            top_radius_ratio: Top radius as fraction of base radius
            
        Returns:
            Conical spiral waypoints
        """
        center_xy, base_radius, height = self._fit_cylinder(pointcloud)
        
        # Conical parameters
        min_alt_offset = getattr(config, "min_flight_altitude_m", None)
        if min_alt_offset is None:
            min_alt_offset = config.altitude * 0.5
        z_start = float(np.min(pointcloud.points[:, 2])) + float(min_alt_offset)
        z_end = float(np.max(pointcloud.points[:, 2]))
        
        # Flight radius at base and top
        global_distance = getattr(config, "global_distance_m", None)
        if global_distance is not None:
            r_base = base_radius + float(global_distance)
        else:
            r_base = base_radius * 1.8
        r_top = r_base * top_radius_ratio
        
        # Spiral parameters
        fov_v_rad = math.radians(camera.fov_vertical_deg)
        pitch = 2 * r_base * math.tan(fov_v_rad / 2) * (1 - config.front_overlap)
        pitch = max(pitch, 2.0)
        
        turns = height / pitch
        total_angle = turns * 2 * math.pi
        num_points = max(int(turns * 20), 50)
        
        cx, cy = center_xy
        waypoints = []
        prev_heading = None
        
        for i in range(num_points):
            t = i / (num_points - 1)
            theta = t * total_angle
            
            # Linear interpolation of radius
            radius = r_base + t * (r_top - r_base)
            
            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)
            z = z_start + t * (z_end - z_start)
            
            dx = cx - x
            dy = cy - y
            heading = self._unwrap_heading_to_previous(
                math.degrees(math.atan2(dx, dy)),
                prev_heading,
            )
            prev_heading = heading
            
            wp = Waypoint(
                x=float(x),
                y=float(y),
                z=float(z),
                heading_deg=heading,
                gimbal_pitch_deg=-45.0,
                speed_ms=config.speed_ms,
                action=WaypointAction.SHOOT,
                dwell_time_s=0.0,
                is_keypoint=True
            )
            waypoints.append(wp)
        
        return waypoints

    @staticmethod
    def _unwrap_heading_to_previous(raw_heading_deg: float, previous_heading_deg: float) -> float:
        """Keep heading continuous to avoid 360/0 wrap discontinuities.

        This prevents interpolation artifacts where one point per revolution can
        swing to the opposite direction when crossing the wrap boundary.
        """
        if previous_heading_deg is None:
            return float(raw_heading_deg)

        candidate = float(raw_heading_deg)
        while candidate - previous_heading_deg > 180.0:
            candidate -= 360.0
        while candidate - previous_heading_deg < -180.0:
            candidate += 360.0
        return candidate
