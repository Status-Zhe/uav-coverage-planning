"""Boustrophedon (S-shaped scan) trajectory generator."""

import math
from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

from ..base_generator import BaseGeometryGenerator, GeneratorConfig
from ..registry import register_generator
from ..coordinate_utils import CoordinateTransformer
from ...models.waypoint import Waypoint, WaypointAction
from ...models.camera import Camera
from ...models.pointcloud import PointCloud


@register_generator("boustrophedon")
class BoustrophedonGenerator(BaseGeometryGenerator):
    """Boustrophedon coverage pattern generator.
    
    Generates a zigzag (S-shaped) scanning pattern for area coverage.
    Optimized for top-down mapping of planar surfaces like roofs or terrain.
    
    Algorithm:
    1. Extract 2D footprint from point cloud
    2. Select longest edge as scan direction
    3. Calculate track spacing based on altitude, FOV, and overlap
    4. Generate parallel scan lines
    5. Connect in zigzag pattern (alternating direction per line)
    
    Reference:
        Boustrophedon Motion Planning for Multi-Robot Coverage
        (Based on traditional coverage path planning)
    """
    
    @property
    def name(self) -> str:
        return "boustrophedon"
    
    def generate(
        self,
        pointcloud: PointCloud,
        camera: Camera,
        config: GeneratorConfig
    ) -> List[Waypoint]:
        """Generate Boustrophedon scanning pattern.
        
        Args:
            pointcloud: Target structure point cloud
            camera: Camera specification
            config: Generator configuration
            
        Returns:
            List of waypoints in zigzag scan order
        """
        if bool(getattr(config, "region_only_enabled", False)):
            rect = getattr(config, "coverage_area_rect_xy", None)
            if rect is None:
                raise ValueError("coverage_area_rect_xy is required in region-only mode")

            xmin, ymin, xmax, ymax = [float(value) for value in rect]
            footprint = Polygon([
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
            ])

            global_distance = getattr(config, "global_distance_m", None)
            if global_distance is None:
                adjusted_altitude = float(config.altitude)
            else:
                adjusted_altitude = max(float(global_distance), float(config.altitude))

            min_alt_offset = getattr(config, "min_flight_altitude_m", None)
            ground_z = float(getattr(config, "region_ground_z", 0.0))
            if min_alt_offset is not None:
                adjusted_altitude = max(adjusted_altitude, ground_z + float(min_alt_offset))

            track_spacing = self._compute_track_spacing(
                adjusted_altitude, camera, config.side_overlap
            )
            scan_angle = self._select_scan_direction(footprint)
            scan_lines = self._generate_scan_lines(footprint, scan_angle, track_spacing)
            return self._lines_to_waypoints(scan_lines, adjusted_altitude, config.speed_ms)

        # Step 0: Detect and handle coordinate system
        transformer = CoordinateTransformer(pointcloud.points)
        
        # Adjust altitude BEFORE transforming (uses original z_min)
        global_distance = getattr(config, "global_distance_m", None)
        if global_distance is None:
            effective_altitude = float(config.altitude)
        else:
            effective_altitude = max(float(global_distance), float(config.altitude))

        min_alt_offset = getattr(config, "min_flight_altitude_m", None)
        if min_alt_offset is not None:
            min_allowed_altitude = float(np.min(pointcloud.points[:, 2])) + float(min_alt_offset)
            effective_altitude = max(effective_altitude, min_allowed_altitude)

        adjusted_altitude = transformer.adjust_altitude(effective_altitude)
        
        # Transform point cloud to local coordinates if needed
        if transformer.is_camera_coords():
            local_points = transformer.to_local(pointcloud.points)
            pointcloud = PointCloud(points=local_points)
        
        # Step 1: Extract footprint polygon from point cloud
        footprint, building_height = self._extract_footprint(pointcloud)

        if adjusted_altitude < building_height:
            print(f"Warning: Desired altitude {adjusted_altitude:.2f} m is below building height {building_height:.2f} m. Adjusting altitude to {building_height + 5:.2f} m.")
            adjusted_altitude = adjusted_altitude + building_height

        # Step 2: Calculate track spacing 注意：使用原始高度减去建筑高度来计算有效飞行高度
        print(f"building_height: {building_height:.2f} m")
        # track_spacing = self._compute_track_spacing(
        #     adjusted_altitude - building_height, camera, config.side_overlap
        # )
        track_spacing = self._compute_track_spacing(
            adjusted_altitude, camera, config.side_overlap
        )  
        # Step 3: Select scan direction (longest edge of footprint)
        scan_angle = self._select_scan_direction(footprint)
        
        # Step 4: Generate scan lines
        scan_lines = self._generate_scan_lines(
            footprint, scan_angle, track_spacing
        )
        
        # Step 5: Generate waypoints from scan lines in zigzag pattern
        waypoints = self._lines_to_waypoints(
            scan_lines, adjusted_altitude, config.speed_ms
        )
        
        # Step 6: Transform waypoints back to original coordinate system
        if transformer.is_camera_coords():
            for wp in waypoints:
                # Convert Z back to original coordinate system
                wp_z_array = np.array([[0, 0, wp.z]])
                wp_z_original = transformer.from_local(wp_z_array)[0, 2]
                wp.z = float(wp_z_original)
        
        return waypoints
    
    def _extract_footprint(self, pointcloud: PointCloud) -> tuple[Polygon, float]:
        """Extract 2D convex hull as footprint polygon, considering building height."""
        from scipy.spatial import ConvexHull
        
        # 获取建筑最高点
        z_max = np.max(pointcloud.points[:, 2])
        z_min = np.min(pointcloud.points[:, 2]) 
        print(f"Point cloud Z range: {z_min:.2f} m to {z_max:.2f} m")        

        # 使用所有点的XY投影（保持不变）
        points_xy = pointcloud.points[:, :2]
        hull = ConvexHull(points_xy)
        hull_points = points_xy[hull.vertices]
        
        # 返回多边形和最高点信息
        polygon = Polygon(hull_points)
        building_height = z_max - z_min  
        
        return polygon, building_height
        
    def _compute_track_spacing(
        self,
        altitude: float,
        camera: Camera,
        side_overlap: float
    ) -> float:
        """Calculate distance between scan lines.
        
        Formula: spacing = 2 * altitude * tan(fov_h/2) * (1 - overlap)
        
        Args:
            altitude: Flight altitude in meters
            camera: Camera specification
            side_overlap: Side overlap ratio (0-1)
            
        Returns:
            Track spacing in meters
        """
        # 计算参数
        fov_h_rad = math.radians(camera.fov_horizontal_deg)
        half_fov_rad = fov_h_rad / 2
        tan_half_fov = math.tan(half_fov_rad)
        ground_width = 2 * altitude * tan_half_fov
        spacing = ground_width * (1 - side_overlap)
        final_spacing = max(spacing, 1.0)
        
        # # 输出所有参数
        # print("=" * 50)
        # print("扫掠间距计算参数:")
        # print("=" * 50)
        # print(f"输入参数:")
        # print(f"  - 飞行高度 (altitude):           {altitude:.2f} m")
        # print(f"  - 相机水平FOV (fov_h):           {camera.fov_horizontal_deg:.1f}°")
        # print(f"  - 旁向重叠率 (side_overlap):     {side_overlap:.2f} ({side_overlap*100:.1f}%)")
        # print()
        # print(f"计算过程:")
        # print(f"  - FOV弧度 (fov_h_rad):           {fov_h_rad:.4f} rad")
        # print(f"  - 半FOV弧度 (half_fov_rad):      {half_fov_rad:.4f} rad")
        # print(f"  - tan(半FOV):                    {tan_half_fov:.4f}")
        # print(f"  - 地面覆盖宽度 (ground_width):   {ground_width:.2f} m")
        # print(f"    公式: 2 × {altitude:.2f} × {tan_half_fov:.4f} = {ground_width:.2f}")
        # print(f"  - 有效覆盖宽度 (spacing):        {spacing:.2f} m")
        # print(f"    公式: {ground_width:.2f} × (1 - {side_overlap:.2f}) = {spacing:.2f}")
        # print()
        # print(f"最终结果:")
        # print(f"  - 扫掠间距 (track_spacing):      {final_spacing:.2f} m")
        # if spacing < 1.0:
        #     print(f"    注: 原始间距 {spacing:.2f}m < 1.0m，已应用最小值限制")
        # print("=" * 50)
        
        return final_spacing
    
    def _select_scan_direction(self, footprint: Polygon) -> float:
        """Select optimal scan direction.
        
        Chooses direction perpendicular to the longest edge
        to minimize number of scan lines.
        
        Args:
            footprint: Footprint polygon
            
        Returns:
            Scan angle in degrees (0 = along x-axis)
        """
        # Get minimum rotated rectangle
        min_rect = footprint.minimum_rotated_rectangle
        coords = list(min_rect.exterior.coords)[:-1]  # Remove duplicate last point
        
        # Find longest edge
        max_length = 0
        best_angle = 0.0
        
        for i in range(len(coords)):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % len(coords)]
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            if length > max_length:
                max_length = length
                best_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # Scan direction is perpendicular to longest edge
        return (best_angle + 90) % 180
    
    def _generate_scan_lines(
        self,
        footprint: Polygon,
        scan_angle: float,
        spacing: float
    ) -> List[LineString]:
        """Generate parallel scan lines across the footprint.
        
        Args:
            footprint: Target polygon
            scan_angle: Scan direction in degrees
            spacing: Distance between lines
            
        Returns:
            List of scan line segments within footprint
        """
        # Rotate footprint to align with scan direction
        angle_rad = math.radians(scan_angle)
        
        # Get bounding box in rotated frame
        bounds = footprint.bounds
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        
        # Transform to rotated coordinate system
        def rotate_point(x, y):
            dx = x - center_x
            dy = y - center_y
            rx = dx * math.cos(-angle_rad) - dy * math.sin(-angle_rad)
            ry = dx * math.sin(-angle_rad) + dy * math.cos(-angle_rad)
            return rx, ry
        
        def unrotate_point(rx, ry):
            x = rx * math.cos(angle_rad) - ry * math.sin(angle_rad) + center_x
            y = rx * math.sin(angle_rad) + ry * math.cos(angle_rad) + center_y
            return x, y
        
        # Get bounds in rotated frame
        rotated_coords = [rotate_point(x, y) for x, y in footprint.exterior.coords]
        min_rx = min(c[0] for c in rotated_coords)
        max_rx = max(c[0] for c in rotated_coords)
        min_ry = min(c[1] for c in rotated_coords)
        max_ry = max(c[1] for c in rotated_coords)
        
        # Generate scan lines in rotated frame
        scan_lines = []
        y = min_ry
        
        while y <= max_ry:
            # Create line across full width
            line = LineString([
                (min_rx - spacing, y),
                (max_rx + spacing, y)
            ])
            
            # Transform back to original frame
            line_original = LineString([
                unrotate_point(x, y) for x, y in line.coords
            ])
            
            # Intersect with footprint
            intersection = line_original.intersection(footprint)
        
            if not intersection.is_empty:
                if intersection.geom_type == 'LineString':
                    scan_lines.append(intersection)
                elif intersection.geom_type == 'MultiLineString':
                    scan_lines.extend(list(intersection.geoms))
            
            y += spacing
        
        # print(f"Generated {len(scan_lines)} scan lines with spacing {spacing:.2f} m at angle {scan_angle:.1f}°")
        return scan_lines
    
    def _lines_to_waypoints(
        self,
        scan_lines: List[LineString],
        altitude: float,
        speed_ms: float
    ) -> List[Waypoint]:
        """Convert scan lines to waypoints in zigzag order.
        
        Args:
            scan_lines: List of scan line segments
            altitude: Flight altitude
            speed_ms: Flight speed
            
        Returns:
            List of waypoints forming zigzag pattern
        """
        waypoints = []
        
        for i, line in enumerate(scan_lines):
            coords = list(line.coords)
            
            # Alternate direction for zigzag pattern
            if i % 2 == 1:
                coords = coords[::-1]
            
            for j, (x, y) in enumerate(coords):
                # Calculate heading (point to next point, or maintain previous)
                if j < len(coords) - 1:
                    next_x, next_y = coords[j + 1]
                    heading = math.degrees(math.atan2(next_x - x, next_y - y))
                elif waypoints:
                    heading = waypoints[-1].heading_deg
                else:
                    heading = 0.0
                
                waypoint = Waypoint(
                    x=float(x),
                    y=float(y),
                    z=altitude,
                    heading_deg=heading,
                    gimbal_pitch_deg=-90.0,  # Nadir (straight down)
                    speed_ms=speed_ms,
                    action=WaypointAction.SHOOT,
                    dwell_time_s=0.0,
                    is_keypoint=True  # All Boustrophedon waypoints are keypoints
                )
                waypoints.append(waypoint)
        
        return waypoints
