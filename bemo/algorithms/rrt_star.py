"""RRT* planner with multi-objective edge cost."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

from bemo.objectives.base import PathResult
from .base import PlannerBase, PlannerConfig

if TYPE_CHECKING:
    from bemo.environment.obstacles import ObstacleMap
    from bemo.environment.uav import UAVDynamics


@dataclass
class RRTNode:
    position: np.ndarray
    parent: Optional[int]   # index into node list; None for root
    cost: float             # cost-from-root
    children: List[int] = field(default_factory=list)


class RRTStarPlanner(PlannerBase):
    """RRT* with scalarized multi-objective edge cost.

    Cost = w_length * edge_length + w_safety * (-min_clearance_along_edge)
    Rewiring ensures asymptotic optimality.
    """

    def __init__(self,
                 config: Optional[PlannerConfig] = None,
                 objective_weights: Optional[Dict[str, float]] = None,
                 step_size: float = 0.3,
                 neighbor_radius: float = 0.8,
                 goal_bias: float = 0.1):
        config = config or PlannerConfig()
        super().__init__(config)
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.goal_bias = goal_bias
        self.weights = objective_weights or {
            "length": 1.0,
            "safety": -2.0,   # negative: more clearance = less cost
        }
        self._nodes: List[RRTNode] = []
        self._rng = np.random.default_rng(config.seed)
        self._kdtree: Optional[KDTree] = None
        self._kdtree_rebuild_freq = 50

    def name(self) -> str:
        return "RRT*"

    def plan(self,
             start: np.ndarray,
             target: np.ndarray,
             obstacle_map: "ObstacleMap",
             bounds: np.ndarray,
             dynamics: "UAVDynamics") -> PathResult:
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        bounds = np.asarray(bounds, dtype=float)

        self._nodes = [RRTNode(start.copy(), parent=None, cost=0.0)]
        self._rebuild_kdtree()

        best_goal_idx: Optional[int] = None
        best_goal_cost = float("inf")
        t0 = time.time()

        for iteration in range(self.config.max_iterations):
            if time.time() - t0 > self.config.timeout_seconds:
                break

            # Rebuild KD-tree periodically
            if iteration % self._kdtree_rebuild_freq == 0 and iteration > 0:
                self._rebuild_kdtree()

            # Sample
            q_rand = self._sample(target, bounds)

            # Nearest node
            q_near_idx = self._nearest(q_rand)
            q_near_pos = self._nodes[q_near_idx].position

            # Steer
            q_new_pos = self._steer(q_near_pos, q_rand)

            # Collision check
            if not obstacle_map.is_collision_free(q_new_pos, self.config.uav_radius):
                continue
            if not self._edge_free(q_near_pos, q_new_pos, obstacle_map):
                continue

            # Find neighbors
            neighbors = self._near(q_new_pos)

            # Choose best parent
            new_node = self._choose_parent(q_new_pos, neighbors, q_near_idx, obstacle_map)
            new_idx = len(self._nodes)
            self._nodes.append(new_node)
            if new_node.parent is not None:
                self._nodes[new_node.parent].children.append(new_idx)

            # Rewire
            self._rewire(new_idx, neighbors, obstacle_map)

            # Check if near goal
            dist_to_goal = float(np.linalg.norm(q_new_pos - target))
            if dist_to_goal < self.step_size * 1.5:
                goal_cost = new_node.cost + self._edge_cost(q_new_pos, target, obstacle_map)
                if goal_cost < best_goal_cost:
                    best_goal_cost = goal_cost
                    best_goal_idx = new_idx

        return self._extract_path(best_goal_idx, target, dynamics, bounds)

    def _sample(self, target: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        if self._rng.random() < self.goal_bias:
            return target.copy()
        low, high = bounds[0], bounds[1]
        return self._rng.uniform(low, high)

    def _nearest(self, q_rand: np.ndarray) -> int:
        if self._kdtree is None or len(self._nodes) == 1:
            return 0
        _, idx = self._kdtree.query(q_rand)
        return int(idx)

    def _steer(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray:
        direction = q_rand - q_near
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            return q_near.copy()
        if dist <= self.step_size:
            return q_rand.copy()
        return q_near + direction / dist * self.step_size

    def _near(self, q_new: np.ndarray) -> List[int]:
        """Find all nodes within neighbor_radius of q_new."""
        if self._kdtree is None:
            return []
        indices = self._kdtree.query_ball_point(q_new, self.neighbor_radius)
        return [int(i) for i in indices]

    def _choose_parent(self, q_new_pos: np.ndarray,
                        neighbors: List[int],
                        default_parent: int,
                        obstacle_map: "ObstacleMap") -> RRTNode:
        """Choose parent that minimizes cost-from-root to q_new."""
        best_parent = default_parent
        best_cost = (self._nodes[default_parent].cost +
                     self._edge_cost(self._nodes[default_parent].position,
                                     q_new_pos, obstacle_map))

        for idx in neighbors:
            if idx == default_parent:
                continue
            node = self._nodes[idx]
            if not self._edge_free(node.position, q_new_pos, obstacle_map):
                continue
            c = node.cost + self._edge_cost(node.position, q_new_pos, obstacle_map)
            if c < best_cost:
                best_cost = c
                best_parent = idx

        return RRTNode(q_new_pos.copy(), parent=best_parent, cost=best_cost)

    def _rewire(self, new_idx: int, neighbors: List[int],
                 obstacle_map: "ObstacleMap") -> None:
        """Rewire: check if routing through new_node improves neighbors' costs."""
        new_node = self._nodes[new_idx]
        for idx in neighbors:
            if idx == new_node.parent:
                continue
            node = self._nodes[idx]
            if not self._edge_free(new_node.position, node.position, obstacle_map):
                continue
            new_cost = (new_node.cost +
                        self._edge_cost(new_node.position, node.position, obstacle_map))
            if new_cost < node.cost - 1e-8:
                # Detach from old parent
                old_parent = node.parent
                if old_parent is not None and idx in self._nodes[old_parent].children:
                    self._nodes[old_parent].children.remove(idx)
                # Attach to new parent
                self._nodes[idx].parent = new_idx
                self._nodes[idx].cost = new_cost
                self._nodes[new_idx].children.append(idx)
                # Propagate cost update to descendants
                self._propagate_cost(idx)

    def _propagate_cost(self, idx: int) -> None:
        """Recursively update costs of all descendants of node idx."""
        visited = {idx}
        stack = list(self._nodes[idx].children)
        while stack:
            child_idx = stack.pop()
            if child_idx in visited:
                continue
            visited.add(child_idx)
            parent_idx = self._nodes[child_idx].parent
            if parent_idx is not None and parent_idx in visited:
                edge_c = self._edge_cost(
                    self._nodes[parent_idx].position,
                    self._nodes[child_idx].position,
                    None  # skip obstacle check for cost propagation
                )
                self._nodes[child_idx].cost = self._nodes[parent_idx].cost + edge_c
            stack.extend(self._nodes[child_idx].children)

    def _edge_cost(self, p1: np.ndarray, p2: np.ndarray,
                    obstacle_map: Optional["ObstacleMap"]) -> float:
        """Weighted multi-objective cost for a single edge.

        Costs are always non-negative to preserve tree validity.
        Safety bonus is applied as a discount on length cost, bounded to [0, length].
        """
        length = float(np.linalg.norm(p2 - p1))
        base_cost = self.weights.get("length", 1.0) * length

        safety_bonus = 0.0
        if obstacle_map is not None and "safety" in self.weights:
            clearance = obstacle_map.segment_clearance(p1, p2, n_samples=10)
            # Negative weight → higher clearance reduces cost. Clamp so cost >= 0.
            safety_bonus = abs(self.weights["safety"]) * clearance

        # Ensure cost is always positive (required for tree validity)
        return max(1e-6, base_cost - safety_bonus)

    def _edge_free(self, p1: np.ndarray, p2: np.ndarray,
                    obstacle_map: "ObstacleMap",
                    n_samples: int = 15) -> bool:
        """Check if entire edge is collision-free."""
        for t in np.linspace(0.0, 1.0, n_samples):
            pt = p1 + t * (p2 - p1)
            if not obstacle_map.is_collision_free(pt, self.config.uav_radius):
                return False
        return True

    def _rebuild_kdtree(self) -> None:
        if len(self._nodes) > 1:
            positions = np.array([n.position for n in self._nodes])
            self._kdtree = KDTree(positions)

    def _extract_path(self, goal_idx: Optional[int], target: np.ndarray,
                       dynamics: "UAVDynamics", bounds: np.ndarray) -> PathResult:
        """Walk parent pointers to get waypoints, simulate dynamics."""
        if goal_idx is None:
            # Return best-effort path toward goal
            goal_idx = self._closest_to_target(target)

        # Trace path from root to goal (with cycle detection)
        path_indices = []
        visited: set = set()
        idx: Optional[int] = goal_idx
        while idx is not None and idx not in visited:
            path_indices.append(idx)
            visited.add(idx)
            idx = self._nodes[idx].parent
        path_indices.reverse()

        waypoints = [self._nodes[i].position for i in path_indices]
        if len(waypoints) == 0:
            waypoints = [self._nodes[0].position]
        # Append target
        waypoints.append(target)

        return self._simulate_along_waypoints(waypoints, dynamics, bounds)

    def _closest_to_target(self, target: np.ndarray) -> int:
        """Find node closest to target."""
        positions = np.array([n.position for n in self._nodes])
        dists = np.linalg.norm(positions - target, axis=1)
        return int(np.argmin(dists))

    def _simulate_along_waypoints(self, waypoints: List[np.ndarray],
                                   dynamics: "UAVDynamics",
                                   bounds: np.ndarray) -> PathResult:
        """Simulate UAV dynamics moving through waypoints."""
        from bemo.environment.uav import UAVState

        if len(waypoints) < 2:
            pos = waypoints[0] if waypoints else np.zeros(3)
            return PathResult(
                waypoints=np.array([pos]),
                velocities=np.zeros((1, 3)),
                accelerations=np.zeros((1, 3)),
                timestamps=np.array([0.0]),
                uav_id=0,
                reached_target=False,
                algorithm=self.name(),
            )

        state = UAVState(
            position=waypoints[0].copy(),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            heading=0.0,
            uav_id=0,
        )

        positions, velocities, accelerations, timestamps = [state.position.copy()], [state.velocity.copy()], [state.acceleration.copy()], [0.0]
        t = 0.0
        wp_idx = 1

        while wp_idx < len(waypoints):
            target_wp = waypoints[wp_idx]
            direction = target_wp - state.position
            dist = float(np.linalg.norm(direction))

            if dist < 0.1:
                wp_idx += 1
                continue

            # Acceleration toward waypoint
            desired_vel = direction / dist * dynamics.max_speed
            action = (desired_vel - state.velocity) / dynamics.dt
            action_norm = np.linalg.norm(action)
            if action_norm > dynamics.max_acceleration:
                action = action / action_norm * dynamics.max_acceleration

            state = dynamics.step(state, action)
            state.position = np.clip(state.position, bounds[0], bounds[1])
            t += dynamics.dt

            positions.append(state.position.copy())
            velocities.append(state.velocity.copy())
            accelerations.append(state.acceleration.copy())
            timestamps.append(t)

            # Break if stuck
            if len(timestamps) > 5000:
                break

        target = waypoints[-1]
        reached = float(np.linalg.norm(positions[-1] - target)) < 0.3

        return PathResult(
            waypoints=np.array(positions),
            velocities=np.array(velocities),
            accelerations=np.array(accelerations),
            timestamps=np.array(timestamps),
            uav_id=0,
            reached_target=reached,
            algorithm=self.name(),
        )
