"""Adaptive voxel-based collision checker with ray casting for internal collision detection."""
import numpy as np
from scipy.spatial import cKDTree

from .base_validator import BaseValidator, ValidationResult


class CollisionChecker(BaseValidator):
    """Collision detection using adaptive voxel grid + ray casting.
    
    Creates a 3D voxel grid from point cloud and checks trajectory
    against occupied voxels with configurable safety margin.
    Also supports ray casting to detect points inside hollow structures.
    """
    
    def __init__(self, pointcloud, voxel_size: float = 1.0, safety_margin: float = 2.0, use_ray_casting: bool = False):
        """Initialize collision checker.
        
        Args:
            pointcloud: Input point cloud
            voxel_size: Voxel grid resolution
            safety_margin: Safety distance around obstacles
            use_ray_casting: If True, enable ray casting for internal collision detection
        """
        self.voxel_size = voxel_size
        self.safety_margin = safety_margin
        self.use_ray_casting = use_ray_casting
        
        self._build_voxel_grid(pointcloud)
    
    def _build_voxel_grid(self, pointcloud):
        """Build voxel grid from point cloud."""
        points = pointcloud.points
        
        # Compute bounds with safety margin
        self.min_bounds = np.min(points, axis=0) - self.safety_margin
        self.max_bounds = np.max(points, axis=0) + self.safety_margin
        
        # Quantize to voxels
        voxel_indices = np.floor((points - self.min_bounds) / self.voxel_size).astype(int)
        
        # Store occupied voxels as set for fast lookup
        self.occupied_voxels = set(map(tuple, voxel_indices))
        
        # Build KD-tree for distance queries
        self.kdtree = cKDTree(points)
    
    def check_collision(self, position: np.ndarray) -> bool:
        """Check if position collides with obstacles.
        
        Args:
            position: 3D position to check
            
        Returns:
            True if collision detected
        """
        # First check: distance-based collision (surface proximity)
        distance, _ = self.kdtree.query(position)
        if distance < self.safety_margin:
            return True
        
        # Second check: ray casting for internal collision (if enabled)
        if self.use_ray_casting:
            # print(f"Checking ray casting collision at position {position}")  # Debug log for ray casting --- IGNORE ---
            if self._check_internal_collision(position):
                # print(f"Internal collision detected at position {position}")  # Debug log for internal collision --- IGNORE ---
                return True
        
        return False
    
    def _check_internal_collision(self, position: np.ndarray) -> bool:
        """Check if point is inside a hollow structure using ray casting.
        
        Uses 5-direction ray casting (up, front, back, left, right).
        Point is considered inside ONLY if ALL rays hit occupied voxels
        in BOTH directions (meaning point is completely enclosed).
        
        Args:
            position: 3D position to check
            
        Returns:
            True if point is inside a hollow structure (all rays hit voxels)
        """
        # Define ray directions (positive and negative along each axis)
        directions = [
            np.array([1.0, 0.0, 0.0]),   # +X
            np.array([-1.0, 0.0, 0.0]),  # -X
            np.array([0.0, 1.0, 0.0]),   # +Y
            np.array([0.0, -1.0, 0.0]),  # -Y
            np.array([0.0, 0.0, 1.0]),   # +Z
        ]
        
        point_voxel = np.floor((position - self.min_bounds) / self.voxel_size).astype(int)
         
        debug_ind = False
        # if (-64.28820572<position[0]<-64.28820570) and (-21.27429289<position[1]<-21.27429287) and (5.01129455<position[2]<5.01129457):
        #     debug_ind = True

        if tuple(point_voxel) in self.occupied_voxels:
            return True

        # Check each direction
        for direction in directions:
            if not self._ray_hits_voxel(position, direction, debug_ind=debug_ind):
                # If any ray doesn't hit a voxel, point is not enclosed
                return False
        
         
        # All rays hit voxels → point is inside enclosed space
        return True
    
    def _ray_hits_voxel(self, origin: np.ndarray, direction: np.ndarray, debug_ind: bool = False, dir_name: str = "") -> bool:
        """Check if a ray hits any occupied voxel before exiting the grid."""
        axis = np.argmax(np.abs(direction))
        step_dir = np.sign(direction).astype(int)
        
        start_voxel = np.floor((origin - self.min_bounds) / self.voxel_size).astype(int)
        if debug_ind:
            print(f"  Ray casting from {origin} in direction {dir_name} (voxel {start_voxel})")
        current_voxel = start_voxel.copy()
        current_voxel[axis] += step_dir[axis]
        
        max_idx = np.floor((self.max_bounds - self.min_bounds) / self.voxel_size).astype(int)
        max_steps = int(np.max(max_idx)) + 5
        

        steps_taken = 0
        for step in range(max_steps):
            # 边界检查
            if np.any(current_voxel < 0) or np.any(current_voxel > max_idx):
                if debug_ind:
                    print(f"  Ray exited grid at step {step}, voxel {current_voxel}")
                break
            
            # 检查占用
            voxel_tuple = tuple(current_voxel)
            if voxel_tuple in self.occupied_voxels:
                if debug_ind:
                    print(f"  Hit occupied voxel at step {step}: {voxel_tuple}")
                return True
            
            # 调试：对于+Z方向，打印前几个检查的体素
            if debug_ind and dir_name == '+Z' and step < 10:
                print(f"  Step {step}: checking voxel {voxel_tuple} - {'occupied' if voxel_tuple in self.occupied_voxels else 'empty'}")
            
            current_voxel[axis] += step_dir[axis]
            steps_taken = step
        
        if debug_ind:
            print(f"  No hit after {steps_taken} steps")
        
        return False
        
    def _voxel_in_bounds(self, voxel: np.ndarray) -> bool:
        """Check if a voxel index is within the grid bounds."""
        min_idx = np.zeros(3, dtype=int)
        max_idx = np.floor((self.max_bounds - self.min_bounds) / self.voxel_size).astype(int)
        
        return np.all(voxel >= min_idx) and np.all(voxel <= max_idx)
    
    def check_trajectory(self, waypoints: list) -> ValidationResult:
        """Check entire trajectory for collisions.
        
        Args:
            waypoints: List of waypoint positions
            
        Returns:
            ValidationResult with collision status
        """
        collisions = []
        
        for i, wp in enumerate(waypoints):
            pos = np.array([wp.x, wp.y, wp.z])
            if self.check_collision(pos):
                collision_info = {
                    'index': i,
                    'position': pos,
                    'distance': self.kdtree.query(pos)[0]
                }
                
                collisions.append(collision_info)
        
        if collisions:
            return ValidationResult(
                valid=False,
                errors=[f"Collision at waypoint {c['index']}" for c in collisions],
                details={'collisions': collisions}
            )
        
        return ValidationResult(valid=True)
    
    def validate(self, waypoints: list) -> list:
        """Implement abstract method from BaseValidator.
        
        Args:
            waypoints: Waypoints to validate
            
        Returns:
            List of bool (True = no collision at that waypoint)
        """
        return [not self.check_collision(np.array([wp.x, wp.y, wp.z])) for wp in waypoints]
    
    def filter_safe_waypoints(self, waypoints: list) -> list:
        """Filter waypoints to keep only safe (non-colliding) ones.
        
        Args:
            waypoints: List of waypoints to filter
            
        Returns:
            List of safe waypoints
        """
        safe = []
        for wp in waypoints:
            pos = np.array([wp.x, wp.y, wp.z])
            if not self.check_collision(pos):
                safe.append(wp)
        return safe