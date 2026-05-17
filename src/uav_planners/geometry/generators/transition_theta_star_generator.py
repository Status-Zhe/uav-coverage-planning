"""Deterministic transition generator based on Theta* + shortcut smoothing.

Goals:
1. Prefer direct straight-line transition when collision-free
2. Use local 3D Theta* search when blocked
3. Remove redundant vertices with line-of-sight shortcutting
4. Densify output with moderate spacing for stable downstream execution
"""

from __future__ import annotations

import heapq
import logging
from itertools import count
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ...models.camera import Camera
from ...models.pointcloud import PointCloud
from ...models.waypoint import Waypoint
from ..base_generator import BaseGeometryGenerator, GeneratorConfig
from ..registry import register_generator

logger = logging.getLogger(__name__)


@register_generator("transition_theta_star")
class TransitionThetaStarGenerator(BaseGeometryGenerator):
    """Stage4 transition generator using deterministic Theta* search."""

    DEFAULT_GRID_RESOLUTION: float = 2.0
    DEFAULT_MAX_EXPANSIONS: int = 30000
    DEFAULT_GOAL_TOLERANCE: float = 2.0
    DEFAULT_WAYPOINT_SPACING: float = 4.0
    DEFAULT_BOUND_MARGIN: float = 20.0

    def __init__(
        self,
        grid_resolution: float = DEFAULT_GRID_RESOLUTION,
        max_expansions: int = DEFAULT_MAX_EXPANSIONS,
        safety_margin: float = 2.0,
        goal_tolerance: float = DEFAULT_GOAL_TOLERANCE,
        waypoint_spacing: float = DEFAULT_WAYPOINT_SPACING,
        shortcut_divisor: float = 20.0,
    ):
        self.grid_resolution = max(0.5, grid_resolution)
        self.max_expansions = max(1000, max_expansions)
        self.safety_margin = max(0.1, safety_margin)
        self.goal_tolerance = max(0.5, goal_tolerance)
        self.waypoint_spacing = max(1.0, waypoint_spacing)
        self.shortcut_divisor = max(2.0, shortcut_divisor)  # Ensure divisor >= 2 to avoid infinite loops
        # print(f"Initialized TransitionThetaStarGenerator with grid_resolution={self.grid_resolution}, "
        #       f"max_expansions={self.max_expansions}, safety_margin={self.safety_margin}, "
        #       f"goal_tolerance={self.goal_tolerance}, waypoint_spacing={self.waypoint_spacing}, "
        #       f"shortcut_divisor={self.shortcut_divisor}")
    @property
    def name(self) -> str:
        return "transition_theta_star"

    def generate(
        self,
        pointcloud: PointCloud,
        camera: Camera,
        config: GeneratorConfig,
        obstacle_tree: Optional[cKDTree] = None,  # Optional cached tree from pipeline
    ) -> List[List[Waypoint]]:
        _ = camera
        paths: List[List[Waypoint]] = []

        if hasattr(config, "transitions") and config.transitions:
            for start, goal in config.transitions:
                paths.append(self._generate_single_path(start, goal, pointcloud, config, obstacle_tree))
            return paths

        if getattr(config, "start_waypoint", None) and getattr(config, "goal_waypoint", None):
            path = self._generate_single_path(config.start_waypoint, config.goal_waypoint, pointcloud, config, obstacle_tree)
            return [path]

        raise ValueError("Config must have either transitions or start_waypoint + goal_waypoint")

    def _generate_single_path(
        self,
        start: Waypoint,
        goal: Waypoint,
        pointcloud: PointCloud,
        config: Optional[GeneratorConfig] = None,
        obstacle_tree: Optional[cKDTree] = None,  # Optional cached tree from pipeline
    ) -> List[Waypoint]:
        start_pos = np.array([start.x, start.y, start.z], dtype=float)
        goal_pos = np.array([goal.x, goal.y, goal.z], dtype=float)

        # Use cached tree if provided (from pipeline), otherwise build new one for backward compatibility
        tree = obstacle_tree if obstacle_tree is not None else (cKDTree(pointcloud.points) if pointcloud.point_count > 0 else None)

        prefer_lateral = bool(getattr(config, "transition_prefer_lateral_before_altitude", True))
        offset_min = float(getattr(config, "transition_lateral_offset_min_m", max(6.0, self.safety_margin * 3.0)))
        offset_max = float(getattr(config, "transition_lateral_offset_max_m", max(30.0, offset_min)))
        offset_step = float(getattr(config, "transition_lateral_offset_step_m", 4.0))
        max_candidates = int(getattr(config, "transition_lateral_max_candidates", 16))
        turn_penalty_weight = float(getattr(config, "transition_lateral_turn_penalty_weight", 0.15))
        enable_theta_star_fallback = bool(getattr(config, "transition_enable_theta_star_fallback", False))

        if self._segment_collision_free(start_pos, goal_pos, tree):
            return self._to_waypoints([start_pos, goal_pos], start, goal)

        preferred_candidates: List[List[np.ndarray]] = []
        fallback_candidates: List[List[np.ndarray]] = []

        if enable_theta_star_fallback:
            theta_candidate = self._theta_star_path(start_pos, goal_pos, pointcloud, tree)
            if theta_candidate:
                preferred_candidates.append(theta_candidate)

        if preferred_candidates == []:
            lateral_candidate = self._lateral_corridor_fallback(
                start_pos,
                goal_pos,
                tree,
                offset_min,
                offset_max,
                offset_step,
                max_candidates,
                turn_penalty_weight,
            )

            if lateral_candidate:
                if prefer_lateral:
                    preferred_candidates.append(lateral_candidate)
                else:
                    fallback_candidates.append(lateral_candidate)

            altitude_candidate = self._altitude_corridor_fallback(start_pos, goal_pos, pointcloud, tree)
            if altitude_candidate:
                if prefer_lateral:
                    fallback_candidates.append(altitude_candidate)
                else:
                    preferred_candidates.append(altitude_candidate)

        candidate_groups = [preferred_candidates, fallback_candidates]
        raw_candidates: List[List[np.ndarray]] = []
        for group in candidate_groups:
            if group:
                raw_candidates = group
                break

        if not raw_candidates:
            raise RuntimeError("Transition planner failed: no collision-free deterministic corridor")

        best_dense_points: Optional[List[np.ndarray]] = None
        best_cost = float("inf")

        for raw_path in raw_candidates:
            simplified = self._shortcut_path(raw_path, tree)
            smoothed = self._smooth_path(simplified, tree)
            dense_points = self._densify_path(smoothed)
            if not dense_points or len(dense_points) < 2:
                continue

            path_cost = self._path_length(dense_points)
            if path_cost < best_cost:
                best_cost = path_cost
                best_dense_points = dense_points

        if best_dense_points is None:
            raise RuntimeError("Transition planner failed: no valid candidate after post-processing")

        return self._to_waypoints(best_dense_points, start, goal)

    def _path_length(self, points: List[np.ndarray]) -> float:
        if len(points) < 2:
            return 0.0

        total = 0.0
        for index in range(len(points) - 1):
            total += float(np.linalg.norm(points[index + 1] - points[index]))
        return total

    def _lateral_corridor_fallback(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        tree: Optional[cKDTree],
        offset_min_m: float,
        offset_max_m: float,
        offset_step_m: float,
        max_candidates: int,
        turn_penalty_weight: float,
    ) -> Optional[List[np.ndarray]]:
        """Try adaptive XY dog-leg corridors and choose the lowest-cost valid one."""
        delta_xy = np.array([goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1]], dtype=float)
        norm_xy = np.linalg.norm(delta_xy)
        if norm_xy < 1e-6:
            return None

        perp = np.array([-delta_xy[1], delta_xy[0]], dtype=float) / norm_xy
        min_offset = max(0.5, float(offset_min_m))
        max_offset = max(min_offset, float(offset_max_m))
        step_offset = max(0.25, float(offset_step_m))
        candidate_values = np.arange(min_offset, max_offset + step_offset * 0.5, step_offset)
        offset_candidates = [float(value) for value in candidate_values]
        if len(offset_candidates) > max_candidates:
            indices = np.linspace(0, len(offset_candidates) - 1, max_candidates).astype(int)
            offset_candidates = [offset_candidates[index] for index in indices]

        best_path: Optional[List[np.ndarray]] = None
        best_cost = float("inf")

        for sign in (1.0, -1.0):
            for offset in offset_candidates:
                shift = np.array([perp[0] * offset * sign, perp[1] * offset * sign, 0.0], dtype=float)
                p1 = start_pos.copy()
                p2 = start_pos + shift
                p3 = goal_pos + shift
                p4 = goal_pos.copy()

                if (
                    self._segment_collision_free(p1, p2, tree)
                    and self._segment_collision_free(p2, p3, tree)
                    and self._segment_collision_free(p3, p4, tree)
                ):
                    candidate = [p1, p2, p3, p4]
                    length = (
                        np.linalg.norm(p2 - p1)
                        + np.linalg.norm(p3 - p2)
                        + np.linalg.norm(p4 - p3)
                    )
                    turn_penalty = abs(offset) * max(0.0, turn_penalty_weight)
                    cost = float(length + turn_penalty)

                    if cost < best_cost:
                        best_cost = cost
                        best_path = candidate

        return best_path

    def _theta_star_path(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        pointcloud: PointCloud,
        tree: Optional[cKDTree],
        expansion_factor: float = 1.0,
        use_global_bounds: bool = False,
    ) -> Optional[List[np.ndarray]]:
        if self._point_in_collision(start_pos, tree) or self._point_in_collision(goal_pos, tree):
            return None

        min_xyz, max_xyz = self._local_bounds(
            start_pos,
            goal_pos,
            pointcloud,
            expansion_factor=expansion_factor,
            use_global_bounds=use_global_bounds,
        )
        origin = min_xyz
        size = np.maximum(max_xyz - min_xyz, self.grid_resolution)
        max_idx = np.ceil(size / self.grid_resolution).astype(int)

        def to_grid(point: np.ndarray) -> Tuple[int, int, int]:
            idx = np.round((point - origin) / self.grid_resolution).astype(int)
            idx = np.clip(idx, 0, max_idx)
            return int(idx[0]), int(idx[1]), int(idx[2])

        def to_world(idx: Tuple[int, int, int]) -> np.ndarray:
            return origin + np.array(idx, dtype=float) * self.grid_resolution

        start_idx = to_grid(start_pos)
        goal_idx = to_grid(goal_pos)

        open_heap: List[Tuple[float, int, Tuple[int, int, int]]] = []
        tie = count()
        heapq.heappush(open_heap, (0.0, next(tie), start_idx))

        g_score: Dict[Tuple[int, int, int], float] = {start_idx: 0.0}
        parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {start_idx: start_idx}
        closed = set()
        blocked_cache: Dict[Tuple[int, int, int], bool] = {}
        los_cache: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], bool] = {}

        best_idx = start_idx
        best_dist = np.linalg.norm(start_pos - goal_pos)
        expansions = 0

        while open_heap and expansions < self.max_expansions:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            expansions += 1

            current_world = to_world(current)
            current_goal_dist = np.linalg.norm(current_world - goal_pos)
            if current_goal_dist < best_dist:
                best_dist = current_goal_dist
                best_idx = current

            if current_goal_dist <= self.goal_tolerance:
                best_idx = current
                break

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
                        if (
                            nxt[0] < 0 or nxt[1] < 0 or nxt[2] < 0
                            or nxt[0] > max_idx[0] or nxt[1] > max_idx[1] or nxt[2] > max_idx[2]
                        ):
                            continue

                        if nxt in blocked_cache:
                            if blocked_cache[nxt]:
                                continue
                        else:
                            is_blocked = self._point_in_collision(to_world(nxt), tree)
                            blocked_cache[nxt] = is_blocked
                            if is_blocked:
                                continue

                        candidate_parent = current
                        candidate_cost = g_score[current] + np.linalg.norm(to_world(nxt) - current_world)

                        parent_current = parent.get(current, current)
                        if parent_current != current:
                            key = (parent_current, nxt) if parent_current <= nxt else (nxt, parent_current)
                            if key in los_cache:
                                has_los = los_cache[key]
                            else:
                                has_los = self._segment_collision_free(to_world(parent_current), to_world(nxt), tree)
                                los_cache[key] = has_los
                            if has_los:
                                candidate_parent = parent_current
                                candidate_cost = g_score[parent_current] + np.linalg.norm(
                                    to_world(nxt) - to_world(parent_current)
                                )

                        if candidate_cost < g_score.get(nxt, float("inf")):
                            g_score[nxt] = candidate_cost
                            parent[nxt] = candidate_parent
                            heuristic = np.linalg.norm(to_world(nxt) - goal_pos)
                            heapq.heappush(open_heap, (candidate_cost + heuristic, next(tie), nxt))

        if best_idx not in parent:
            return None

        idx_path = [best_idx]
        cur = best_idx
        while cur != start_idx:
            cur = parent[cur]
            idx_path.append(cur)
        idx_path.reverse()

        world_path = [to_world(idx) for idx in idx_path]
        world_path[0] = start_pos

        if np.linalg.norm(world_path[-1] - goal_pos) <= self.goal_tolerance:
            world_path[-1] = goal_pos
            return world_path

        if self._segment_collision_free(world_path[-1], goal_pos, tree):
            world_path.append(goal_pos)
            return world_path

        return None

    def _altitude_corridor_fallback(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        pointcloud: PointCloud,
        tree: Optional[cKDTree],
    ) -> Optional[List[np.ndarray]]:
        """Deterministic fallback: climb, translate, descend with collision checks."""
        if tree is None:
            return [start_pos, goal_pos]

        base_top = max(start_pos[2], goal_pos[2]) + max(8.0, self.safety_margin * 4.0)
        if pointcloud.point_count > 0:
            base_top = max(base_top, pointcloud.bounds.max_z + self.safety_margin * 2.0)

        candidate_heights = [base_top, base_top + 8.0, base_top + 16.0]
        for z in candidate_heights:
            p1 = start_pos.copy()
            p2 = np.array([start_pos[0], start_pos[1], z], dtype=float)
            p3 = np.array([goal_pos[0], goal_pos[1], z], dtype=float)
            p4 = goal_pos.copy()
            if (
                self._segment_collision_free(p1, p2, tree)
                and self._segment_collision_free(p2, p3, tree)
                and self._segment_collision_free(p3, p4, tree)
            ):
                return [p1, p2, p3, p4]

        return None

    def _shortcut_path(self, points: List[np.ndarray], tree: Optional[cKDTree]) -> List[np.ndarray]:
        """
        Aggressive bidirectional shortcutting algorithm.
        
        Strategy:
        - Sample path points at intervals of gapstep = n / shortcut_divisor
        - Try connecting non-adjacent points from both ends toward center
        - If direct connection is collision-free, delete all intermediate points
        - Update length dynamically after each successful shortcut
        """
        if len(points) <= 1:
            return list(points)
        
        n = len(points)
        gapstep = max(1, int(n / self.shortcut_divisor))  # TODO #2: Calculate gap size based on divisor
        
        # Calculate maximum iterations to prevent index out-of-bounds (TODO #3)
        # Condition: i < j means gapstep*times < n - gapstep*times
        # => 2 * gapstep * times < n
        # => times < n / (2 * gapstep)
        max_times = min((n - gapstep * 2) // gapstep + 1, n // gapstep)
        
        if max_times <= 0:
            return list(points)
        
        result = list(points)  # Work on a copy
        
        # Bidirectional aggressive shortcutting (TODO #2)
        for times in range(1, max_times):
            i = gapstep * times           # Forward index from start
            j = n - gapstep * times       # Backward index from end
            
            if i >= j:
                break
            
            # Attempt direct connection between distant points (skipping all intermediate points)
            if self._segment_collision_free(result[i], result[j], tree):
                # Delete all intermediate points and update length dynamically (TODO #2)
                del result[i+1:j]
                n = len(result)  # Critical: Update length after deletion to prevent index errors
        
        return result



    def _smooth_path(self, points: List[np.ndarray], tree: Optional[cKDTree], passes: int = 1) -> List[np.ndarray]:

        if len(points) <= 2:
            return points

        smoothed = [point.copy() for point in points]
        for _ in range(passes):
            updated = [smoothed[0]]
            for index in range(1, len(smoothed) - 1):
                prev_point = updated[-1]
                curr_point = smoothed[index]
                next_point = smoothed[index + 1]
                candidate = (prev_point + curr_point + next_point) / 3.0
                if (
                    not self._point_in_collision(candidate, tree)
                    and self._segment_collision_free(prev_point, candidate, tree)
                    and self._segment_collision_free(candidate, next_point, tree)
                ):
                    updated.append(candidate)
                else:
                    updated.append(curr_point)
            updated.append(smoothed[-1])
            smoothed = updated

        return smoothed

    def _densify_path(self, points: List[np.ndarray]) -> List[np.ndarray]:
        if len(points) <= 1:
            return points

        dense: List[np.ndarray] = [points[0]]
        for index in range(len(points) - 1):
            start_point = points[index]
            end_point = points[index + 1]
            segment_length = np.linalg.norm(end_point - start_point)
            steps = max(1, int(np.ceil(segment_length / self.waypoint_spacing)))
            for step in range(1, steps + 1):
                t = step / steps
                dense.append(start_point + t * (end_point - start_point))

        return dense

    def _to_waypoints(self, points: List[np.ndarray], start: Waypoint, goal: Waypoint) -> List[Waypoint]:
        if len(points) == 1:
            points = [points[0], points[0]]

        waypoints: List[Waypoint] = []
        total = len(points)
        for i, point in enumerate(points):
            t = i / max(1, total - 1)
            if i < total - 1:
                diff = points[i + 1] - point
                heading = float(np.degrees(np.arctan2(diff[0], diff[1])))
            else:
                heading = goal.heading_deg

            waypoint = Waypoint(
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]),
                heading_deg=heading,
                gimbal_pitch_deg=float(start.gimbal_pitch_deg + t * (goal.gimbal_pitch_deg - start.gimbal_pitch_deg)),
                speed_ms=float(start.speed_ms + t * (goal.speed_ms - start.speed_ms)),
                action=start.action if i < total - 1 else goal.action,
                dwell_time_s=0.0,
                is_keypoint=(i == 0 or i == total - 1),
                waypoint_type="transition",
            )
            waypoints.append(waypoint)

        return waypoints

    def _point_in_collision(self, point: np.ndarray, tree: Optional[cKDTree]) -> bool:
        if tree is None:
            return False
        distance, _ = tree.query(point, k=1)
        return float(distance) < self.safety_margin

    def _segment_collision_free(
        self,
        start: np.ndarray,
        end: np.ndarray,
        tree: Optional[cKDTree],
    ) -> bool:
        if tree is None:
            return True

        length = np.linalg.norm(end - start)
        samples = max(2, int(np.ceil(length / max(self.grid_resolution, 1.5))))
        for i in range(samples + 1):
            t = i / samples
            point = start + t * (end - start)
            if self._point_in_collision(point, tree):
                return False
        return True

    def _local_bounds(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        pointcloud: PointCloud,
        expansion_factor: float = 1.0,
        use_global_bounds: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if use_global_bounds and pointcloud.point_count > 0:
            margin = max(self.DEFAULT_BOUND_MARGIN, self.safety_margin * 4.0) * expansion_factor
            bounds = pointcloud.bounds
            return (
                np.array([bounds.min_x - margin, bounds.min_y - margin, 0.0], dtype=float),
                np.array([bounds.max_x + margin, bounds.max_y + margin, bounds.max_z + margin], dtype=float),
            )

        min_xyz = np.minimum(start_pos, goal_pos)
        max_xyz = np.maximum(start_pos, goal_pos)

        span = np.linalg.norm(goal_pos - start_pos)
        margin = max(self.DEFAULT_BOUND_MARGIN, span * 0.5, self.safety_margin * 4.0) * expansion_factor
        min_xyz = min_xyz - margin
        max_xyz = max_xyz + margin
        min_xyz[2] = max(0.0, min_xyz[2])

        if pointcloud.point_count > 0:
            bounds = pointcloud.bounds
            global_min = np.array([bounds.min_x - margin, bounds.min_y - margin, 0.0], dtype=float)
            global_max = np.array([bounds.max_x + margin, bounds.max_y + margin, bounds.max_z + margin], dtype=float)
            min_xyz = np.maximum(min_xyz, global_min)
            max_xyz = np.minimum(max_xyz, global_max)

        return min_xyz, max_xyz