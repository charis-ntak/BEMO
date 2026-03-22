"""APF-PSO planner: Artificial Potential Fields + Particle Swarm Optimization."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from bemo.objectives.base import PathResult
from .base import PlannerBase, PlannerConfig

if TYPE_CHECKING:
    from bemo.environment.obstacles import ObstacleMap, Obstacle
    from bemo.environment.uav import UAVDynamics


@dataclass
class Particle:
    """PSO particle encoding a waypoint sequence."""
    position: np.ndarray   # shape (n_waypoints * 3,)
    velocity: np.ndarray   # shape (n_waypoints * 3,)
    best_position: np.ndarray
    best_fitness: float


class APFPSOPlanner(PlannerBase):
    """Artificial Potential Field + Particle Swarm Optimization planner.

    Phase 1: APF gradient descent produces initial path.
    Phase 2: PSO refines waypoints to improve multi-objective cost.
    Unknown obstacles are discovered progressively within sensor_radius.
    """

    def __init__(self,
                 config: Optional[PlannerConfig] = None,
                 n_particles: int = 30,
                 n_waypoints: int = 15,
                 sensor_radius: float = 2.0,
                 k_att: float = 1.0,
                 k_rep: float = 2.0,
                 d0: float = 1.5,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 pso_iterations: int = 100):
        config = config or PlannerConfig()
        super().__init__(config)
        self.n_particles = n_particles
        self.n_waypoints = n_waypoints
        self.sensor_radius = sensor_radius
        self.k_att = k_att
        self.k_rep = k_rep
        self.d0 = d0
        self.w = w       # inertia
        self.c1 = c1     # cognitive
        self.c2 = c2     # social
        self.pso_iterations = pso_iterations
        self._rng = np.random.default_rng(config.seed)

    def name(self) -> str:
        return "APF-PSO"

    def plan(self,
             start: np.ndarray,
             target: np.ndarray,
             obstacle_map: "ObstacleMap",
             bounds: np.ndarray,
             dynamics: "UAVDynamics") -> PathResult:
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        bounds = np.asarray(bounds, dtype=float)

        # Phase 1: APF gradient descent with progressive obstacle discovery
        apf_waypoints = self._apf_path(start, target, obstacle_map, bounds, dynamics)

        # Phase 2: PSO refinement
        optimized_waypoints = self._pso_optimize(apf_waypoints, target, obstacle_map, bounds)

        # Simulate dynamics along optimized waypoints
        return self._simulate_along_waypoints(
            [start] + list(optimized_waypoints) + [target], dynamics, bounds, obstacle_map
        )

    def _apf_path(self, start: np.ndarray, target: np.ndarray,
                   obstacle_map: "ObstacleMap",
                   bounds: np.ndarray,
                   dynamics: "UAVDynamics",
                   max_steps: int = 500) -> List[np.ndarray]:
        """APF gradient descent with sensor-limited obstacle discovery."""
        pos = start.copy()
        waypoints = [pos.copy()]
        known_ids: Set[int] = set()
        discovered: List["Obstacle"] = []
        stagnation_count = 0
        prev_dist = float(np.linalg.norm(target - pos))

        step_size = dynamics.max_speed * dynamics.dt * 5  # larger steps for planning

        for _ in range(max_steps):
            # Discover obstacles within sensor radius
            nearby = obstacle_map.obstacles_within_radius(pos, self.sensor_radius)
            for obs in nearby:
                if obs.id not in known_ids:
                    known_ids.add(obs.id)
                    discovered.append(obs)

            # Compute APF gradient
            force = self._apf_force(pos, target, discovered)
            force_norm = float(np.linalg.norm(force))

            if force_norm < 1e-6:
                break

            # Move in force direction
            pos = pos + (force / force_norm) * step_size
            pos = np.clip(pos, bounds[0], bounds[1])
            waypoints.append(pos.copy())

            dist = float(np.linalg.norm(target - pos))
            if dist < step_size:
                break

            # Stagnation detection
            if abs(prev_dist - dist) < 0.01:
                stagnation_count += 1
                if stagnation_count > 20:
                    # Escape: random perturbation
                    pos += self._rng.uniform(-0.5, 0.5, size=3)
                    pos = np.clip(pos, bounds[0], bounds[1])
                    stagnation_count = 0
            else:
                stagnation_count = 0
            prev_dist = dist

        return waypoints

    def _apf_force(self, position: np.ndarray, target: np.ndarray,
                    discovered: List["Obstacle"]) -> np.ndarray:
        """Total APF force = attractive toward goal + repulsive from obstacles."""
        # Attractive force
        att = self.k_att * (target - position)

        # Repulsive forces from discovered obstacles
        rep = np.zeros(3)
        for obs in discovered:
            d = obs.sdf(position)
            if d <= 0:
                d = 1e-3  # inside obstacle: strong repulsion
            if d < self.d0:
                grad = self._numerical_gradient(obs.sdf, position)
                coeff = self.k_rep * (1.0 / d - 1.0 / self.d0) * (1.0 / d**2)
                rep -= coeff * grad

        return att + rep

    def _numerical_gradient(self, sdf_fn, point: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        grad = np.zeros(3)
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = eps
            grad[i] = (sdf_fn(point + dp) - sdf_fn(point - dp)) / (2 * eps)
        return grad

    def _pso_optimize(self, apf_waypoints: List[np.ndarray],
                       target: np.ndarray,
                       obstacle_map: "ObstacleMap",
                       bounds: np.ndarray) -> np.ndarray:
        """PSO over waypoint space. Returns optimized interior waypoints."""
        # Sample n_waypoints evenly from APF path as initial solution
        n = self.n_waypoints
        indices = np.linspace(0, len(apf_waypoints) - 1, n).astype(int)
        initial_wps = np.array([apf_waypoints[i] for i in indices])  # shape (n, 3)

        dim = n * 3
        low = np.tile(bounds[0], n)
        high = np.tile(bounds[1], n)

        # Initialize particles
        particles: List[Particle] = []
        for i in range(self.n_particles):
            if i == 0:
                pos = initial_wps.flatten()
            else:
                pos = self._rng.uniform(low, high)
            vel = self._rng.uniform(-0.5, 0.5, size=dim)
            fit = self._particle_fitness(pos.reshape(n, 3), target, obstacle_map)
            particles.append(Particle(pos.copy(), vel.copy(), pos.copy(), fit))

        # Global best
        g_best_pos = min(particles, key=lambda p: p.best_fitness).best_position.copy()
        g_best_fit = min(particles, key=lambda p: p.best_fitness).best_fitness

        for _ in range(self.pso_iterations):
            for p in particles:
                r1 = self._rng.uniform(0, 1, size=dim)
                r2 = self._rng.uniform(0, 1, size=dim)
                p.velocity = (self.w * p.velocity
                              + self.c1 * r1 * (p.best_position - p.position)
                              + self.c2 * r2 * (g_best_pos - p.position))
                p.position = np.clip(p.position + p.velocity, low, high)

                fit = self._particle_fitness(p.position.reshape(n, 3), target, obstacle_map)
                if fit < p.best_fitness:
                    p.best_fitness = fit
                    p.best_position = p.position.copy()
                if fit < g_best_fit:
                    g_best_fit = fit
                    g_best_pos = p.position.copy()

        return g_best_pos.reshape(n, 3)

    def _particle_fitness(self, waypoints: np.ndarray,
                           target: np.ndarray,
                           obstacle_map: "ObstacleMap") -> float:
        """Evaluate waypoint sequence: path_length + collision_penalty - clearance."""
        total_length = 0.0
        min_clearance = float("inf")
        collision_penalty = 0.0

        prev = waypoints[0] if len(waypoints) > 0 else target
        for wp in waypoints:
            total_length += float(np.linalg.norm(wp - prev))
            c = obstacle_map.clearance(wp)
            min_clearance = min(min_clearance, c)
            if c < self.config.uav_radius:
                collision_penalty += 10.0
            prev = wp
        total_length += float(np.linalg.norm(target - prev))

        clearance_bonus = max(0.0, min_clearance) * 2.0
        return total_length + collision_penalty - clearance_bonus

    def _simulate_along_waypoints(self, waypoints: List[np.ndarray],
                                   dynamics: "UAVDynamics",
                                   bounds: np.ndarray,
                                   obstacle_map: "ObstacleMap") -> PathResult:
        """Simulate UAV dynamics moving through waypoints."""
        from bemo.environment.uav import UAVState

        if not waypoints:
            return PathResult(
                waypoints=np.zeros((1, 3)),
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

        positions = [state.position.copy()]
        velocities = [state.velocity.copy()]
        accelerations = [state.acceleration.copy()]
        timestamps = [0.0]
        n_collisions = 0
        t = 0.0
        wp_idx = 1

        while wp_idx < len(waypoints) and len(timestamps) < 5000:
            target_wp = waypoints[wp_idx]
            direction = target_wp - state.position
            dist = float(np.linalg.norm(direction))

            if dist < 0.1:
                wp_idx += 1
                continue

            desired_vel = direction / dist * dynamics.max_speed * 0.8
            action = (desired_vel - state.velocity) / dynamics.dt
            a_norm = float(np.linalg.norm(action))
            if a_norm > dynamics.max_acceleration:
                action = action / a_norm * dynamics.max_acceleration

            state = dynamics.step(state, action)
            state.position = np.clip(state.position, bounds[0], bounds[1])
            t += dynamics.dt

            if not obstacle_map.is_collision_free(state.position, self.config.uav_radius):
                n_collisions += 1

            positions.append(state.position.copy())
            velocities.append(state.velocity.copy())
            accelerations.append(state.acceleration.copy())
            timestamps.append(t)

        target = waypoints[-1]
        reached = float(np.linalg.norm(positions[-1] - target)) < 0.3

        return PathResult(
            waypoints=np.array(positions),
            velocities=np.array(velocities),
            accelerations=np.array(accelerations),
            timestamps=np.array(timestamps),
            uav_id=0,
            reached_target=reached,
            n_collisions=n_collisions,
            algorithm=self.name(),
        )
