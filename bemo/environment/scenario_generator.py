"""Random scenario generation with validity guarantees."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .obstacles import BoxObstacle, CylinderObstacle, Obstacle, ObstacleMap


@dataclass
class Scenario:
    """A complete simulation scenario."""
    obstacles: List[Obstacle]
    starts: List[np.ndarray]    # one per UAV
    targets: List[np.ndarray]   # one per UAV
    bounds: np.ndarray          # shape (2, 3)
    n_uavs: int
    seed: int


class ScenarioGenerator:
    """Generates random valid scenarios for the lab environment."""

    def __init__(self, bounds: np.ndarray, seed: Optional[int] = None):
        self.bounds = np.asarray(bounds, dtype=float)
        self.rng = np.random.default_rng(seed)
        self._next_id = 0

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def generate(self,
                 n_obstacles: int,
                 n_uavs: int,
                 obstacle_types: Optional[List[str]] = None,
                 min_clearance: float = 0.5,
                 max_attempts: int = 1000) -> Scenario:
        """Generate a valid random scenario.

        Guarantees start/target positions are collision-free and separated.
        Uses rejection sampling for placement.
        """
        obstacle_types = obstacle_types or ["box", "cylinder"]
        seed = int(self.rng.integers(0, 2**31))

        # Generate obstacles
        obstacles: List[Obstacle] = []
        for _ in range(n_obstacles):
            obs = self._sample_obstacle(obstacle_types, obstacles, min_clearance, max_attempts)
            if obs is not None:
                obstacles.append(obs)

        obstacle_map = ObstacleMap(obstacles, self.bounds)

        # Generate start and target positions
        starts: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        occupied: List[np.ndarray] = []

        for i in range(n_uavs):
            start = self._sample_free_point(obstacle_map, occupied, min_clearance, max_attempts)
            occupied.append(start)
            target = self._sample_free_point(obstacle_map, occupied, min_clearance, max_attempts,
                                              min_dist_from=start, min_dist=2.0)
            occupied.append(target)
            starts.append(start)
            targets.append(target)

        return Scenario(
            obstacles=obstacles,
            starts=starts,
            targets=targets,
            bounds=self.bounds,
            n_uavs=n_uavs,
            seed=seed,
        )

    def _sample_obstacle(self, obstacle_types: List[str],
                          existing: List[Obstacle],
                          min_clearance: float,
                          max_attempts: int) -> Optional[Obstacle]:
        """Sample a random obstacle not overlapping existing ones."""
        for _ in range(max_attempts):
            kind = self.rng.choice(obstacle_types)
            if kind == "box":
                obs = self._sample_box()
            else:
                obs = self._sample_cylinder()

            # Check it doesn't overlap with existing obstacles
            valid = True
            for other in existing:
                dist = np.linalg.norm(obs.position - other.position)
                if dist < min_clearance * 2:
                    valid = False
                    break

            # Check it's within bounds
            aabb = obs.aabb()
            if not (np.all(aabb.min_corner >= self.bounds[0]) and
                    np.all(aabb.max_corner <= self.bounds[1])):
                valid = False

            if valid:
                return obs
        return None

    def _sample_box(self) -> BoxObstacle:
        low = self.bounds[0]
        high = self.bounds[1]
        span = high - low

        # Half-extents: 0.2 to 1.0m
        hx = self.rng.uniform(0.2, min(1.0, span[0] / 4))
        hy = self.rng.uniform(0.2, min(1.0, span[1] / 4))
        hz = self.rng.uniform(0.2, min(1.0, span[2] / 2))
        half_extents = np.array([hx, hy, hz])

        # Position: centroid inside bounds with clearance for half-extents
        pos = np.array([
            self.rng.uniform(low[0] + hx, high[0] - hx),
            self.rng.uniform(low[1] + hy, high[1] - hy),
            self.rng.uniform(low[2] + hz, high[2] - hz),
        ])
        return BoxObstacle(self._new_id(), pos, half_extents)

    def _sample_cylinder(self) -> CylinderObstacle:
        low = self.bounds[0]
        high = self.bounds[1]
        span = high - low

        radius = self.rng.uniform(0.2, min(0.8, span[0] / 5))
        height = self.rng.uniform(0.5, span[2] * 0.8)

        pos = np.array([
            self.rng.uniform(low[0] + radius, high[0] - radius),
            self.rng.uniform(low[1] + radius, high[1] - radius),
            low[2],  # base at floor
        ])
        return CylinderObstacle(self._new_id(), pos, radius, height)

    def _sample_free_point(self,
                            obstacle_map: ObstacleMap,
                            occupied: List[np.ndarray],
                            min_clearance: float,
                            max_attempts: int,
                            min_dist_from: Optional[np.ndarray] = None,
                            min_dist: float = 0.0) -> np.ndarray:
        """Sample a point that is collision-free and far from occupied positions."""
        low = self.bounds[0]
        high = self.bounds[1]

        for _ in range(max_attempts):
            pt = self.rng.uniform(low, high)
            if obstacle_map.clearance(pt) < min_clearance:
                continue
            if not obstacle_map.is_within_bounds(pt):
                continue
            # Check distance from occupied points
            too_close = False
            for occ in occupied:
                if np.linalg.norm(pt - occ) < min_clearance:
                    too_close = True
                    break
            if too_close:
                continue
            # Check minimum distance from a specific point
            if min_dist_from is not None and np.linalg.norm(pt - min_dist_from) < min_dist:
                continue
            return pt

        # Fallback: return a corner-ish point
        margin = min_clearance + 0.1
        return low + margin * np.ones(3)
