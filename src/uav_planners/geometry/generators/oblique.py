"""Oblique photography (5-direction) trajectory generator with OBB support."""

import math
from typing import List, Tuple, Dict, Optional
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import rotate, translate
from scipy.spatial import ConvexHull

from ..base_generator import BaseGeometryGenerator, GeneratorConfig
from ..registry import register_generator
from ..generators.boustrophedon import BoustrophedonGenerator
from ..coordinate_utils import CoordinateTransformer
from ...models.waypoint import Waypoint, WaypointAction
from ...models.camera import Camera
from ...models.pointcloud import PointCloud


@register_generator("oblique")
class ObliqueGenerator(BaseGeometryGenerator):
    """Oblique photography (5-direction) pattern generator with OBB support.
    
    Generates 5-directional coverage pattern for 3D reconstruction:
    - Top face (nadir, -90°)
    - Front face (oblique, camera pointing to front)
    - Back face (oblique, camera pointing to back)
    - Left face (oblique, camera pointing to left)
    - Right face (oblique, camera pointing to right)
    
    Uses Oriented Bounding Box (OBB) to align with building's principal axes.
    """
    
    def __init__(self):
        self.boustrophedon = BoustrophedonGenerator()
    
    @property
    def name(self) -> str:
        return "oblique"
    
    def generate(
        self,
        pointcloud: PointCloud,
        camera: Camera,
        config: GeneratorConfig
    ) -> List[List[Waypoint]]:
        """Generate 5-direction oblique photography pattern.
        
        Args:
            pointcloud: Target structure point cloud
            camera: Camera specification
            config: Generator configuration
            
        Returns:
            Combined waypoints from all 5 faces
        """
        
        # Step 1: Coordinate transformation for altitude adjustment only
        transformer = CoordinateTransformer(pointcloud.points)
        adjusted_altitude = transformer.adjust_altitude(config.altitude)
        
        # Step 2: Extract OBB from original point cloud (before any transformation)
        obb = self._extract_oriented_bounding_box(pointcloud)
        
        # Step 3: Extract projection polygons for each face
        face_polygons = self._extract_face_polygons(pointcloud, obb)

        # Step 4: Calculate flight parameters USING THE POLYGONS
        # 无人机距离点云的距离在个函数被决定
        # 关于距离还需要调整
        flight_params = self._calculate_flight_parameters_obb(
            obb, face_polygons, camera, config, adjusted_altitude
        )

        # Define face configurations: (face_name, generator_func, is_side)
        face_configs = ["top","front", "back","left","right"]

        waypointslist = []
        for face_name in face_configs:
            params = flight_params[face_name]
            wps = self._generate_face_mission(
                obb, 
                params,             # 飞行参数（包含距离、姿态、扫描平面等）投影多边形
                config,
                face_name
            )

            waypointslist.append(wps)

        return waypointslist
    
    def _extract_oriented_bounding_box(self, pointcloud: PointCloud) -> Dict:
        """Compute OBB using PCA direction + convex hull bounding (guaranteed to contain all points).
        
        Steps:
        1. PCA on XY plane to get principal directions (rotation)
        2. Project ALL points onto PCA axes to find true extents
        3. Build OBB corners from actual min/max projections (not symmetric)
        
        Args:
            pointcloud: Input point cloud
            
        Returns:
            Dictionary with OBB parameters that strictly contains all points
        """
        import numpy as np
        from scipy.spatial import ConvexHull
        import math
        
        points = pointcloud.points
        
        # ========== 1. Z bounds (unchanged) ==========
        z_min = float(np.min(points[:, 2]))
        z_max = float(np.max(points[:, 2]))
        height = max(z_max - z_min, 0.1)
        center_z = (z_min + z_max) / 2
        
        # ========== 2. PCA on XY plane for orientation ==========
        xy_points = points[:, :2]
        
        # Use convex hull vertices only for PCA (more stable, ignores interior points)
        try:
            hull = ConvexHull(xy_points)
            hull_points = xy_points[hull.vertices]
        except:
            # Fallback if convex hull fails (degenerate case)
            hull_points = xy_points
        
        # PCA on hull vertices
        center_xy = np.mean(hull_points, axis=0)
        centered = hull_points - center_xy
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (largest first)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Principal axis (X' axis)
        principal_axis = eigenvectors[:, 0]
        
        # Ensure principal_axis points to positive X direction
        if principal_axis[0] < 0:
            principal_axis = -principal_axis
        
        # Secondary axis (Y' axis) - right-hand rule: Z × X'
        secondary_axis = np.array([principal_axis[1], -principal_axis[0]])
        
        # Build rotation matrix (local to global)
        rotation_matrix = np.array([principal_axis, secondary_axis])  # 2x2 matrix
        
        # ========== 3. Project ALL points onto PCA axes for TRUE extents ==========
        # This is the key: use ALL points, not just hull, to guarantee containment
        
        centered_all = xy_points - center_xy
        proj_principal = np.dot(centered_all, principal_axis)   # X' coordinates
        proj_secondary = np.dot(centered_all, secondary_axis)   # Y' coordinates
        
        # True extents (guaranteed to contain all points)
        min_principal = np.min(proj_principal)
        max_principal = np.max(proj_principal)
        min_secondary = np.min(proj_secondary)
        max_secondary = np.max(proj_secondary)
        
        width = max_principal - min_principal
        depth = max_secondary - min_secondary
        
        # ========== 4. Build OBB corners from TRUE extents ==========
        # Local coordinates: origin at PCA center, but corners at actual min/max
        
        # 8 corners in local (PCA) coordinates
        local_corners = np.array([
            [min_principal, min_secondary, -height/2],
            [max_principal, min_secondary, -height/2],
            [max_principal, max_secondary, -height/2],
            [min_principal, max_secondary, -height/2],
            [min_principal, min_secondary,  height/2],
            [max_principal, min_secondary,  height/2],
            [max_principal, max_secondary,  height/2],
            [min_principal, max_secondary,  height/2]
        ])
        
        # Transform to global coordinates: global = center + R @ local_xy
        global_xy = np.dot(local_corners[:, :2], rotation_matrix) + center_xy
        global_corners = np.hstack([global_xy, local_corners[:, 2:3] + center_z])
        
        # ========== 5. Compute true center (midpoint of extents) ==========
        center_principal = (min_principal + max_principal) / 2
        center_secondary = (min_secondary + max_secondary) / 2
        center_local = np.array([center_principal, center_secondary])
        center_xy_final = np.dot(center_local, rotation_matrix) + center_xy
        
        # ========== 6. Rotation angle ==========
        angle_rad = math.atan2(principal_axis[1], principal_axis[0])
        angle_deg = math.degrees(angle_rad)
        
        return {
            'center_x': float(center_xy_final[0]),
            'center_y': float(center_xy_final[1]),
            'center_z': float(center_z),
            'width': float(width),
            'depth': float(depth),
            'height': float(height),
            'half_width': width / 2,
            'half_depth': depth / 2,
            'half_height': height / 2,
            'angle_deg': angle_deg,
            'angle_rad': angle_rad,
            'principal_axis': principal_axis,
            'secondary_axis': secondary_axis,
            'rotation_matrix': rotation_matrix,
            'corners': global_corners,
            'z_min': z_min,
            'z_max': z_max,
            # Additional info for verification
            'min_principal': float(min_principal),
            'max_principal': float(max_principal),
            'min_secondary': float(min_secondary),
            'max_secondary': float(max_secondary),
        }
    
    def _extract_face_polygons(self, pointcloud: PointCloud, obb: Dict) -> Dict[str, Polygon]:
        """Extract projection polygons for each face of the building.
        
        Args:
            pointcloud: Original point cloud
            obb: Oriented bounding box parameters
            
        Returns:
            Dictionary with polygons for top, front, back, left, right faces
        """
        points = pointcloud.points

        # Transform points to OBB local coordinates
        local_points = self._global_to_local_points(points, obb)

        # Top face: project to XY plane (X', Y')
        top_polygon = self._extract_footprint_2d(local_points[:, :2])
        
        # Front/Back faces: project to X'Z plane
        front_back_points = local_points[:, [0, 2]]  # X' and Z
        front_back_polygon = self._extract_footprint_2d(front_back_points)
        
        # Left/Right faces: project to Y'Z plane
        left_right_points = local_points[:, [1, 2]]  # Y' and Z
        left_right_polygon = self._extract_footprint_2d(left_right_points)
        
        # print(f"  Top polygon area: {top_polygon.area:.2f} m²")
        # print(f"  Front/Back polygon area: {front_back_polygon.area:.2f} m²")
        # print(f"  Left/Right polygon area: {left_right_polygon.area:.2f} m²")
        
        return {
            'top': top_polygon,
            'front': front_back_polygon,
            'back': front_back_polygon,  # Same projection, different orientation
            'left': left_right_polygon,
            'right': left_right_polygon   # Same projection, different orientation
        }

    def _calculate_flight_parameters_obb(
        self, 
        obb: Dict, 
        face_polygons: Dict[str, Polygon],
        camera: Camera, 
        config: GeneratorConfig,
        altitude: float
    ) -> Dict[str, Dict]:
        """Calculate flight parameters based on actual projection polygons.
        
        Args:
            obb: Oriented bounding box parameters
            face_polygons: Projection polygons for each face
            camera: Camera specification
            config: Generator configuration
            altitude: User-specified flight altitude
            
        Returns:
            Flight parameters for each face
        """
        # Safety margin
        safety_margin = float(getattr(config, "global_distance_m", None) or config.oblique_dst_srf)

        # 最低飞行高度限制：统一参数优先，旧参数回退
        min_alt_offset = getattr(config, "min_flight_altitude_m", None)
        if min_alt_offset is None:
            min_alt_offset = config.oblique_min_altitude
        min_flight_altitude = obb['z_min'] + float(min_alt_offset)
        
        # ========== Top Face ==========
        top_polygon = face_polygons['top']
        min_x, min_y, max_x, max_y = top_polygon.bounds
        
        top_plane_z = max(obb['z_max'] + safety_margin, min_flight_altitude)

        # ========== Front/Back Faces (XZ平面) ==========
        xz_polygon = self._clip_polygon_by_height(
            face_polygons['front'], 
            min_flight_altitude,
            axis='z'
        )
        min_xz_x, min_xz_z, max_xz_x, max_xz_z = xz_polygon.bounds

        # ========== Left/Right Faces (YZ平面) ==========
        yz_polygon = self._clip_polygon_by_height(
            face_polygons['left'], 
            min_flight_altitude,
            axis='z'
        )
        min_yz_y, min_yz_z, max_yz_y, max_yz_z = yz_polygon.bounds
        
        # Calculate offsets
        front_offset = -obb['depth']/2 - safety_margin
        back_offset = obb['depth']/2 + safety_margin
        left_offset = -obb['width']/2 - safety_margin
        right_offset = obb['width']/2 + safety_margin
        
        return {
            'top': {
                'distance': safety_margin,
                'plane_z': top_plane_z,
                'gimbal_pitch': -90.0,
                'heading_local': 0.0,
                'scan_plane': 'xy',  # Add missing scan_plane for top face
                'polygon': top_polygon,  # Store polygon for later use
                'bounds': (min_x, min_y, max_x, max_y)
            },
            'front': {
                'distance': safety_margin,
                'plane_coord': front_offset,
                'gimbal_pitch': 0,
                'heading_local': 0.0,
                'scan_plane': 'xz',
                'polygon': xz_polygon,  # Store projection polygon
                'bounds': (min_xz_x, min_xz_z, max_xz_x, max_xz_z)
            },
            'back': {
                'distance': safety_margin,
                'plane_coord': back_offset,
                'gimbal_pitch': 0,
                'heading_local': 180.0,
                'scan_plane': 'xz',
                'polygon': xz_polygon,
                'bounds': (min_xz_x, min_xz_z, max_xz_x, max_xz_z)
            },
            'left': {
                'distance': safety_margin,
                'plane_coord': left_offset,
                'gimbal_pitch': 0,
                'heading_local': -90.0,
                'scan_plane': 'yz',
                'polygon': yz_polygon,
                'bounds': (min_yz_y, min_yz_z, max_yz_y, max_yz_z)
            },
            'right': {
                'distance': safety_margin,
                'plane_coord': right_offset,
                'gimbal_pitch': 0,
                'heading_local': 90.0,
                'scan_plane': 'yz',
                'polygon': yz_polygon,
                'bounds': (min_yz_y, min_yz_z, max_yz_y, max_yz_z)
            }
        }

    def _generate_face_mission(
        self,
        obb: Dict,
        flight_param: Dict,
        config: GeneratorConfig,
        face_name: str
    ) -> List[Waypoint]:
        """统一的航点生成方法，适用于所有面。
        
        Args:
            obb: OBB参数（用于坐标转换）
            polygon: 该面的投影多边形（在局部坐标系中）
            flight_param: 飞行参数（包含距离、姿态、扫描平面等）
            config: 全局配置
            face_name: 面名称（用于日志）
            
        Returns:
            全局坐标系下的航点列表
        """
        polygon = flight_param["polygon"]
        
        # 检查多边形是否有效
        if polygon.is_empty or polygon.area < 0.01:
            print(f"Warning: {face_name} face has no valid coverage area after height clipping, skipping")
            return []
        
        if flight_param['scan_plane'] == 'xy':  # Top面：在X'Y'平面扫描
            waypoints = self._generate_xy_plane_mission(flight_param["polygon"], flight_param, config)
        elif flight_param['scan_plane'] == 'xz': # Front/Back面：在X'Z平面扫描
            waypoints = self._generate_xz_plane_mission(flight_param["polygon"], flight_param, config)
        else:   # Left/Right面：在Y'Z平面扫描
            waypoints = self._generate_yz_plane_mission(flight_param["polygon"], flight_param, config)
        
        global_waypoints = []
        principal_axis = np.asarray(obb['principal_axis'], dtype=np.float64)
        secondary_axis = np.asarray(obb['secondary_axis'], dtype=np.float64)

        # Build inward-looking local normal for side faces so aircraft heading
        # points toward the OBB (not just a fixed angle offset).
        local_inward_dir: Optional[np.ndarray] = None
        if flight_param['scan_plane'] == 'xz':
            # xz scan means y' is fixed plane coord: negative=>front, positive=>back
            # inward should point toward y'=0
            if flight_param['plane_coord'] < 0:
                local_inward_dir = np.array([0.0, 1.0], dtype=np.float64)
            else:
                local_inward_dir = np.array([0.0, -1.0], dtype=np.float64)
        elif flight_param['scan_plane'] == 'yz':
            # yz scan means x' is fixed plane coord: negative=>left, positive=>right
            # inward should point toward x'=0
            if flight_param['plane_coord'] < 0:
                local_inward_dir = np.array([1.0, 0.0], dtype=np.float64)
            else:
                local_inward_dir = np.array([-1.0, 0.0], dtype=np.float64)

        global_inward_heading: Optional[float] = None
        if local_inward_dir is not None:
            inward_global = (
                local_inward_dir[0] * principal_axis
                + local_inward_dir[1] * secondary_axis
            )
            global_inward_heading = float(math.degrees(math.atan2(inward_global[0], inward_global[1])) % 360.0)

        for wp in waypoints:
            global_x, global_y = self._local_to_global_point(wp.x, wp.y, obb)

            # Side faces: heading follows OBB inward normal.
            # Top face: keep historical heading behavior.
            if global_inward_heading is not None:
                global_heading = global_inward_heading
            else:
                global_heading = (flight_param['heading_local'] + obb['angle_deg']) % 360.0
            
            global_waypoints.append(Waypoint(
                x=float(global_x),
                y=float(global_y),
                z=wp.z,
                heading_deg=global_heading,
                gimbal_pitch_deg=flight_param['gimbal_pitch'],
                speed_ms=config.speed_ms,
                action=WaypointAction.SHOOT,
                dwell_time_s=0.0,
                is_keypoint=True
            ))

        return global_waypoints

    """各个平面航点生成函数（基于不同的扫描平面）"""
    def _generate_xy_plane_mission(
        self,
        polygon: Polygon,
        flight_param: Dict,
        config: GeneratorConfig
    ) -> List[Waypoint]:
        """在XY平面（顶面）生成扫描线（局部坐标）。
        
        Args:
            polygon: 投影多边形（在X'Y'平面）
            flight_param: 飞行参数
            config: 全局配置
            
        Returns:
            局部坐标系中的航点列表
        """
        # 计算扫描间距（基于相机FOV和飞行高度）
        fov_h_rad = math.radians(config.fov_horizontal_deg if hasattr(config, 'fov_horizontal_deg') else 60)
        track_spacing = 2 * flight_param['distance'] * math.tan(fov_h_rad / 2) * (1 - config.side_overlap)
        track_spacing = max(track_spacing, 0.5)
        
        # print(f"    Track spacing: {track_spacing:.2f}m")
        
        # 生成扫描线（在X'Y'平面）
        scan_lines = self._generate_scan_lines_from_polygon(polygon, track_spacing)
        
        # print(f"    Number of scan lines: {len(scan_lines)}")
        
        # 转换为航点（之字形，固定Z高度）
        waypoints = []
        for i, line in enumerate(scan_lines):
            coords = list(line.coords)
            # 交替方向实现之字形
            if i % 2 == 1:
                coords = coords[::-1]
            
            for x, y in coords:
                waypoints.append(Waypoint(
                    x=x,
                    y=y,
                    z=flight_param['plane_z'],  # 固定飞行高度
                    heading_deg=flight_param['heading_local'],
                    gimbal_pitch_deg=flight_param['gimbal_pitch'],
                    speed_ms=config.speed_ms,
                    action=WaypointAction.SHOOT,
                    dwell_time_s=0.0,
                    is_keypoint=True
                ))
        
        return waypoints

    def _generate_xz_plane_mission(
        self,
        polygon: Polygon,
        flight_param: Dict,
        config: GeneratorConfig
    ) -> List[Waypoint]:
        """在XZ平面（前后侧面）生成扫描线（局部坐标）。
        
        Args:
            polygon: 投影多边形（在X'Z平面）
            flight_param: 飞行参数
            config: 全局配置
            
        Returns:
            局部坐标系中的航点列表
        """
        # 计算扫描间距（基于相机FOV和飞行距离）
        fov_v_rad = math.radians(config.fov_vertical_deg if hasattr(config, 'fov_vertical_deg') else 45)
        track_spacing = 2 * flight_param['distance'] * math.tan(fov_v_rad / 2) * (1 - config.side_overlap)
        track_spacing = max(track_spacing, 0.5)
        
        # print(f"    Track spacing: {track_spacing:.2f}m")
        
        # 生成扫描线（在X'Z平面）
        scan_lines = self._generate_scan_lines_from_polygon(polygon, track_spacing)
        
        # print(f"    Number of scan lines: {len(scan_lines)}")
        
        # 转换为航点（之字形，固定Y'坐标）
        waypoints = []
        for i, line in enumerate(scan_lines):
            coords = list(line.coords)
            # 交替方向实现之字形
            if i % 2 == 1:
                coords = coords[::-1]
            
            for x, z in coords:
                waypoints.append(Waypoint(
                    x=x,
                    y=flight_param['plane_coord'],  # 固定的Y'坐标
                    z=z,
                    heading_deg=flight_param['heading_local'],
                    gimbal_pitch_deg=flight_param['gimbal_pitch'],
                    speed_ms=config.speed_ms,
                    action=WaypointAction.SHOOT,
                    dwell_time_s=0.0,
                    is_keypoint=True
                ))
        
        return waypoints

    def _generate_yz_plane_mission(
        self,
        polygon: Polygon,
        flight_param: Dict,
        config: GeneratorConfig
    ) -> List[Waypoint]:
        """在YZ平面（左右侧面）生成扫描线（局部坐标）。
        
        Args:
            polygon: 投影多边形（在Y'Z平面）
            flight_param: 飞行参数
            config: 全局配置
            
        Returns:
            局部坐标系中的航点列表
        """
        # 计算扫描间距（基于相机FOV和飞行距离）
        fov_v_rad = math.radians(config.fov_vertical_deg if hasattr(config, 'fov_vertical_deg') else 45)
        track_spacing = 2 * flight_param['distance'] * math.tan(fov_v_rad / 2) * (1 - config.side_overlap)
        track_spacing = max(track_spacing, 0.5)
        
        # print(f"    Track spacing: {track_spacing:.2f}m")
        
        # 生成扫描线（在Y'Z平面）
        scan_lines = self._generate_scan_lines_from_polygon(polygon, track_spacing)
        
        # print(f"    Number of scan lines: {len(scan_lines)}")
        
        # 转换为航点（之字形，固定X'坐标）
        waypoints = []
        for i, line in enumerate(scan_lines):
            coords = list(line.coords)
            # 交替方向实现之字形
            if i % 2 == 1:
                coords = coords[::-1]
            
            for y, z in coords:
                waypoints.append(Waypoint(
                    x=flight_param['plane_coord'],  # 固定的X'坐标
                    y=y,
                    z=z,
                    heading_deg=flight_param['heading_local'],
                    gimbal_pitch_deg=flight_param['gimbal_pitch'],
                    speed_ms=config.speed_ms,
                    action=WaypointAction.SHOOT,
                    dwell_time_s=0.0,
                    is_keypoint=True
                ))
        
        return waypoints

    def _generate_scan_lines_from_polygon(self, polygon: Polygon, spacing: float) -> List[LineString]:
        """从多边形生成水平扫描线（在2D平面中）。
        
        算法：
        1. 获取多边形的边界框
        2. 从下到上生成水平扫描线
        3. 每条扫描线与多边形求交
        4. 返回有效线段
        
        Args:
            polygon: 2D多边形（在任意平面中）
            spacing: 扫描线间距（米）
            
        Returns:
            扫描线段列表（LineString对象）
        """
        if polygon.is_empty or polygon.area == 0:
            print(f"    Warning: Empty polygon, skipping scan line generation")
            return []
        
        min_x, min_y, max_x, max_y = polygon.bounds
        
        # If spacing is larger than polygon height, adjust to ensure at least 5 scan lines
        height = max_y - min_y
        min_scan_lines = 5
        threshold = height / (min_scan_lines - 1)
        if spacing > threshold:
            spacing = threshold
        
        scan_lines = []
        y = min_y
        
        line_count = 0
        while y <= max_y + 0.001:  # Small epsilon for float comparison
            # 创建水平扫描线（稍超出边界以确保相交）
            line = LineString([(min_x - spacing, y), (max_x + spacing, y)])
            
            # 与多边形求交
            intersection = line.intersection(polygon)
            
            if not intersection.is_empty:
                if intersection.geom_type == 'LineString':
                    scan_lines.append(intersection)
                    line_count += 1
                elif intersection.geom_type == 'MultiLineString':
                    for geom in intersection.geoms:
                        scan_lines.append(geom)
                        line_count += 1
                # Point and other types are skipped (tangent intersections)
            
            y += spacing
        
        # Fallback: ensure at least one scan line through center
        if line_count == 0 and not polygon.is_empty:
            center_y = (min_y + max_y) / 2
            line = LineString([(min_x - 1, center_y), (max_x + 1, center_y)])
            intersection = line.intersection(polygon)
            if not intersection.is_empty:
                if intersection.geom_type == 'LineString':
                    scan_lines.append(intersection)
                elif intersection.geom_type == 'MultiLineString':
                    scan_lines.extend(list(intersection.geoms))
        
        return scan_lines
             
    def _extract_footprint_2d(self, points_2d: np.ndarray) -> Polygon:
        """Extract 2D footprint polygon from projected points.
        
        Uses convex hull to create a polygon representing the footprint
        of the point cloud projection.
        
        Args:
            points_2d: 2D projected points, shape (N, 2)
            
        Returns:
            Shapely Polygon representing the footprint
        """
        if len(points_2d) < 3:
            # Not enough points for a polygon, return bounding box
            min_x, min_y = np.min(points_2d, axis=0)
            max_x, max_y = np.max(points_2d, axis=0)
            return Polygon([
                (min_x, min_y), (max_x, min_y), 
                (max_x, max_y), (min_x, max_y)
            ])
        
        try:
            # Use convex hull to get footprint
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]
            return Polygon(hull_points)
        except Exception:
            # Fallback to bounding box if convex hull fails
            min_x, min_y = np.min(points_2d, axis=0)
            max_x, max_y = np.max(points_2d, axis=0)
            return Polygon([
                (min_x, min_y), (max_x, min_y), 
                (max_x, max_y), (min_x, max_y)
            ])

    """辅助函数 - 坐标转换"""

    def _local_to_global_point(
        self, 
        local_x: float, 
        local_y: float,
        obb: Dict
    ) -> Tuple[float, float]:
        """Convert a single point from OBB local coordinates to global coordinates.
        
        Args:
            local_x: X' coordinate in OBB local system (along principal axis)
            local_y: Y' coordinate in OBB local system (along secondary axis)
            obb: OBB parameters containing center, principal_axis, secondary_axis
            
        Returns:
            Global (x, y) coordinates
        """
        # 局部坐标 -> 全局坐标: global = R * local + center
        global_x = obb['center_x'] + local_x * obb['principal_axis'][0] + local_y * obb['secondary_axis'][0]
        global_y = obb['center_y'] + local_x * obb['principal_axis'][1] + local_y * obb['secondary_axis'][1]
        return global_x, global_y

    def _global_to_local_points(self, points: np.ndarray, obb: Dict) -> np.ndarray:
        """Transform points from global coordinates to OBB local coordinates.
        
        Args:
            points: Points in global coordinates, shape (N, 3)
            obb: OBB parameters containing center, principal_axis, secondary_axis
            
        Returns:
            Points in OBB local coordinates (X', Y', Z)
        """
        # 平移
        translated = points.copy()
        translated[:, 0] -= obb['center_x']
        translated[:, 1] -= obb['center_y']
        
        # 旋转矩阵 (从局部到全局的旋转)
        rot_to_global = np.array([
            [obb['principal_axis'][0], obb['secondary_axis'][0], 0],
            [obb['principal_axis'][1], obb['secondary_axis'][1], 0],
            [0, 0, 1]
        ])
        
        # 从全局到局部：使用旋转矩阵的转置（因为是正交矩阵）
        # 局部坐标 = R^T * (全局坐标 - 中心)
        local_points = np.dot(translated, rot_to_global.T)
        
        return local_points

    def _local_to_global_points(self, local_points: np.ndarray, obb: Dict) -> np.ndarray:
        """Transform multiple points from OBB local coordinates to global coordinates.
        
        Args:
            local_points: Points in OBB local coordinates (X', Y', Z), shape (N, 3)
            obb: OBB parameters containing center, principal_axis, secondary_axis
            
        Returns:
            Points in global coordinates, shape (N, 3)
        """
        # 构建旋转矩阵 (从局部到全局)
        rot_to_global = np.array([
            [obb['principal_axis'][0], obb['secondary_axis'][0], 0],
            [obb['principal_axis'][1], obb['secondary_axis'][1], 0],
            [0, 0, 1]
        ])
        
        # 转换XY坐标
        local_xy = local_points[:, :2]
        global_xy = np.dot(local_xy, rot_to_global[:2, :2].T)
        
        # 平移并保持Z坐标
        global_points = np.zeros_like(local_points)
        global_points[:, 0] = global_xy[:, 0] + obb['center_x']
        global_points[:, 1] = global_xy[:, 1] + obb['center_y']
        global_points[:, 2] = local_points[:, 2]  # Z坐标不变
        
        return global_points
    
    def _clip_polygon_by_height(
        self, 
        polygon: Polygon, 
        min_height: float,
        axis: str = 'z'
    ) -> Polygon:
        """裁剪多边形，只保留高于指定高度的部分。
        
        Args:
            polygon: 输入多边形（XZ或YZ平面的投影）
            min_height: 最低高度限制
            axis: 高度轴，'z' 表示Z轴是高度方向
            
        Returns:
            裁剪后的多边形，保证所有点的高度 >= min_height
            如果原始多边形整体低于 min_height，返回一个退化的多边形（空或最小高度处的线）
        """
        from shapely.geometry import box
        
        if polygon.is_empty:
            return polygon
            
        min_x, min_y, max_x, max_y = polygon.bounds
        
        # 如果多边形已经完全高于最低高度，无需裁剪
        if min_y >= min_height:
            return polygon
        
        # 如果多边形整体低于最低高度，无法生成有效航线
        # 返回一个空多边形，让调用方处理（比如跳过这个面）
        if max_y < min_height:
            return Polygon()  # 返回空多边形
        
        # 创建裁剪框：保留 y >= min_height 的部分
        clip_box = box(min_x - 1.0, min_height, max_x + 1.0, max_y + 1.0)
        
        clipped = polygon.intersection(clip_box)
        
        # 确保结果有效
        if clipped.is_empty or clipped.area < 0.01:
            # 裁剪后面积太小，尝试返回最低高度处的水平线
            # 这样至少能生成一条航线在最低高度
            center_x = (min_x + max_x) / 2
            from shapely.geometry import LineString
            line = LineString([(min_x, min_height), (max_x, min_height)])
            # 与多边形求交，获取有效线段
            intersection = line.intersection(polygon)
            if not intersection.is_empty:
                # 用缓冲把线变成窄多边形，确保后续处理能识别
                return intersection.buffer(0.1)
            return Polygon()
        
        return clipped