"""Enhanced obstacle avoidance system for UAV high-altitude operations.

This module provides:
- Multi-layer safety zones
- Dynamic obstacle support
- Real-time warning mechanisms
- Path re-planning interfaces
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon

logger = logging.getLogger(__name__)


class ObstacleType(Enum):
    """Types of obstacles."""
    STATIC = "static"           # Static building/structure
    DYNAMIC = "dynamic"         # Moving object
    TEMPORARY = "temporary"     # Temporary obstacle (scaffolding, crane)
    RESTRICTED = "restricted"   # Restricted airspace


class SafetyLevel(Enum):
    """Safety level for different zones."""
    WARNING = "warning"         # Caution zone
    SAFE = "safe"              # Safe operating distance
    CRITICAL = "critical"       # Immediate stop required


@dataclass
class Obstacle:
    """Represents an obstacle in 3D space."""
    id: str
    position: np.ndarray        # 3D center position
    radius: float                # Influence radius
    type: ObstacleType = ObstacleType.STATIC
    velocity: Optional[np.ndarray] = None  # For dynamic obstacles
    safety_distance: float = 2.0  # Additional safety distance
    
    def get_safe_distance(self) -> float:
        """Get total safe distance (radius + safety_distance)."""
        return self.radius + self.safety_distance
    
    def distance_to(self, position: np.ndarray) -> float:
        """Calculate distance to a position."""
        return np.linalg.norm(position - self.position)
    
    def safety_level(self, position: np.ndarray) -> SafetyLevel:
        """Determine safety level for a position relative to this obstacle."""
        dist = self.distance_to(position)
        safe_dist = self.get_safe_distance()
        
        if dist < safe_dist * 0.5:
            return SafetyLevel.CRITICAL
        elif dist < safe_dist:
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.SAFE


@dataclass
class SafetyZone:
    """Multi-layer safety zone around obstacles."""
    obstacle: Obstacle
    inner_radius: float        # Critical zone
    outer_radius: float         # Warning zone
    
    def contains(self, position: np.ndarray) -> Tuple[bool, SafetyLevel]:
        """Check if position is within this safety zone."""
        dist = self.obstacle.distance_to(position)
        
        if dist < self.inner_radius:
            return True, SafetyLevel.CRITICAL
        elif dist < self.outer_radius:
            return True, SafetyLevel.WARNING
        else:
            return False, SafetyLevel.SAFE


@dataclass
class ObstacleWarning:
    """Warning about potential obstacle collision."""
    obstacle_id: str
    distance: float
    safety_level: SafetyLevel
    position: np.ndarray
    suggested_action: str
    timestamp: float = 0.0


class DynamicObstacleManager:
    """Manages dynamic obstacles for real-time tracking."""
    
    def __init__(
        self,
        prediction_horizon: float = 5.0,  # seconds
        max_velocity: float = 10.0,  # m/s
    ):
        """Initialize dynamic obstacle manager.
        
        Args:
            prediction_horizon: How far ahead to predict obstacle positions
            max_velocity: Maximum expected obstacle velocity
        """
        self.prediction_horizon = prediction_horizon
        self.max_velocity = max_velocity
        self._tracked_obstacles: Set[str] = set()
        self._obstacle_history: dict = {}
    
    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add or update a tracked obstacle."""
        self._tracked_obstacles.add(obstacle.id)
        
        if obstacle.id not in self._obstacle_history:
            self._obstacle_history[obstacle.id] = []
        
        self._obstacle_history[obstacle.id].append({
            "position": obstacle.position.copy(),
            "velocity": obstacle.velocity.copy() if obstacle.velocity is not None else None,
        })
        
        # Keep only recent history
        if len(self._obstacle_history[obstacle.id]) > 100:
            self._obstacle_history[obstacle.id] = self._obstacle_history[obstacle.id][-100:]
    
    def predict_position(self, obstacle_id: str, time_ahead: float) -> Optional[np.ndarray]:
        """Predict obstacle position at a future time.
        
        Args:
            obstacle_id: ID of obstacle to predict
            time_ahead: Seconds ahead to predict
        
        Returns:
            Predicted position or None if not enough data
        """
        if obstacle_id not in self._obstacle_history:
            return None
        
        history = self._obstacle_history[obstacle_id]
        if len(history) < 2:
            return None
        
        # Use velocity from most recent position
        recent = history[-1]
        if recent["velocity"] is None:
            return recent["position"].copy()
        
        # Linear prediction
        predicted = recent["position"] + recent["velocity"] * time_ahead
        
        return predicted
    
    def predict_trajectory_collision(
        self,
        obstacle_id: str,
        trajectory: List[np.ndarray],
        time_per_waypoint: float,
    ) -> Optional[Tuple[int, float]]:
        """Check if trajectory will collide with predicted obstacle position.
        
        Args:
            obstacle_id: ID of obstacle
            trajectory: List of trajectory waypoints
            time_per_waypoint: Time between waypoints
        
        Returns:
            Tuple of (waypoint_index, distance) or None if no collision predicted
        """
        for i, waypoint in enumerate(trajectory):
            time_ahead = i * time_per_waypoint
            predicted_pos = self.predict_position(obstacle_id, time_ahead)
            
            if predicted_pos is None:
                continue
            
            dist = np.linalg.norm(waypoint - predicted_pos)
            
            # Check collision (assuming some radius)
            if dist < 3.0:  # 3m collision radius
                return i, dist
        
        return None


class EnhancedObstacleAvoidance:
    """Enhanced obstacle avoidance system with multi-layer safety.
    
    Integrates with existing CollisionChecker for static obstacles
    and adds dynamic obstacle handling and warning mechanisms.
    """
    
    def __init__(
        self,
        base_collision_checker=None,  # Existing CollisionChecker
        safety_margin: float = 3.0,
        warning_margin: float = 5.0,
        critical_margin: float = 2.0,
    ):
        """Initialize enhanced obstacle avoidance.
        
        Args:
            base_collision_checker: Existing collision checker for static obstacles
            safety_margin: Safe operating distance
            warning_margin: Distance to trigger warning
            critical_margin: Distance for critical alert
        """
        self.base_checker = base_collision_checker
        self.safety_margin = safety_margin
        self.warning_margin = warning_margin
        self.critical_margin = critical_margin
        
        self._dynamic_manager = DynamicObstacleManager()
        self._obstacles: List[Obstacle] = []
        self._warning_callbacks: List[Callable[[ObstacleWarning], None]] = []
        self._emergency_stop_callbacks: List[Callable[[], None]] = []
    
    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add an obstacle to track."""
        self._obstacles.append(obstacle)
        
        if obstacle.type == ObstacleType.DYNAMIC:
            self._dynamic_manager.add_obstacle(obstacle)
    
    def register_warning_callback(
        self, 
        callback: Callable[[ObstacleWarning], None]
    ) -> None:
        """Register a callback for warning events."""
        self._warning_callbacks.append(callback)
    
    def register_emergency_stop(
        self,
        callback: Callable[[], None]
    ) -> None:
        """Register an emergency stop callback."""
        self._emergency_stop_callbacks.append(callback)
    
    def check_position(
        self,
        position: np.ndarray,
        check_dynamic: bool = True,
    ) -> Tuple[bool, List[ObstacleWarning]]:
        """Check if position is safe from all obstacles.
        
        Args:
            position: 3D position to check
            check_dynamic: Whether to check dynamic obstacles
        
        Returns:
            Tuple of (is_safe, list of warnings)
        """
        warnings: List[ObstacleWarning] = []
        is_collision = False
        
        # Check static obstacles (using base checker)
        if self.base_checker is not None:
            if self.base_checker.check_collision(position):
                warnings.append(ObstacleWarning(
                    obstacle_id="static",
                    distance=0.0,
                    safety_level=SafetyLevel.CRITICAL,
                    position=position,
                    suggested_action="EMERGENCY STOP - Static collision",
                ))
                is_collision = True
        
        # Check tracked obstacles
        for obstacle in self._obstacles:
            dist = obstacle.distance_to(position)
            safe_dist = obstacle.get_safe_distance()
            
            if dist < self.critical_margin:
                warnings.append(ObstacleWarning(
                    obstacle_id=obstacle.id,
                    distance=dist,
                    safety_level=SafetyLevel.CRITICAL,
                    position=position,
                    suggested_action="EMERGENCY STOP",
                ))
                is_collision = True
            elif dist < self.warning_margin:
                warnings.append(ObstacleWarning(
                    obstacle_id=obstacle.id,
                    distance=dist,
                    safety_level=SafetyLevel.WARNING,
                    position=position,
                    suggested_action="Reduce speed and maintain distance",
                ))
        
        # Fire warning callbacks
        for warning in warnings:
            for callback in self._warning_callbacks:
                callback(warning)
            
            if warning.safety_level == SafetyLevel.CRITICAL:
                for callback in self._emergency_stop_callbacks:
                    callback()
        
        return not is_collision, warnings
    
    def check_trajectory(
        self,
        trajectory: List[np.ndarray],
        time_per_waypoint: float = 1.0,
    ) -> Tuple[bool, List[ObstacleWarning], List[int]]:
        """Check if trajectory is safe.
        
        Args:
            trajectory: List of trajectory waypoints
            time_per_waypoint: Time between waypoints
        
        Returns:
            Tuple of (is_safe, warnings, collision_indices)
        """
        warnings: List[ObstacleWarning] = []
        collision_indices: List[int] = []
        is_safe = True
        
        for i, waypoint in enumerate(trajectory):
            safe, waypoint_warnings = self.check_position(waypoint)
            
            if not safe:
                collision_indices.append(i)
                is_safe = False
            
            warnings.extend(waypoint_warnings)
        
        return is_safe, warnings, collision_indices
    
    def suggest_avoidance_maneuver(
        self,
        current_position: np.ndarray,
        target_position: np.ndarray,
        obstacles: List[Obstacle],
    ) -> np.ndarray:
        """Suggest an avoidance maneuver to avoid obstacles.
        
        Args:
            current_position: Current UAV position
            target_position: Target position
            obstacles: List of nearby obstacles
        
        Returns:
            Suggested new position
        """
        # Calculate direction to target
        direction = target_position - current_position
        distance = np.linalg.norm(direction)
        
        if distance < 0.01:
            return current_position.copy()
        
        direction = direction / distance
        
        # Check for obstacles in path
        avoidance_vector = np.zeros(3)
        
        for obstacle in obstacles:
            dist = obstacle.distance_to(current_position)
            if dist < self.warning_margin * 2:
                # Calculate avoidance direction (perpendicular)
                to_obstacle = obstacle.position - current_position
                
                # Project onto plane perpendicular to travel direction
                perp_component = to_obstacle - np.dot(to_obstacle, direction) * direction
                
                if np.linalg.norm(perp_component) > 0.01:
                    perp_direction = perp_component / np.linalg.norm(perp_component)
                    # Scale avoidance based on proximity
                    avoidance_magnitude = self.warning_margin * 2 - dist
                    avoidance_vector += perp_direction * avoidance_magnitude
        
        # Combine with original direction
        step_size = min(2.0, distance)  # Don't overshoot
        suggested = current_position + direction * step_size + avoidance_vector * 0.5
        
        return suggested
    
    def get_safety_zones(self, position: np.ndarray) -> List[SafetyZone]:
        """Get all safety zones that contain a position."""
        zones = []
        
        for obstacle in self._obstacles:
            zone = SafetyZone(
                obstacle=obstacle,
                inner_radius=self.critical_margin,
                outer_radius=self.warning_margin,
            )
            
            contains, _ = zone.contains(position)
            if contains:
                zones.append(zone)
        
        return zones
