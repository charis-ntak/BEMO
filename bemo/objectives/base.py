"""Base classes for objective functions and path representation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bemo.environment.obstacles import ObstacleMap


@dataclass
class PathResult:
    """Unified path representation passed to all objective functions."""
    waypoints: np.ndarray       # shape (T, 3) — position at each timestep
    velocities: np.ndarray      # shape (T, 3)
    accelerations: np.ndarray   # shape (T, 3)
    timestamps: np.ndarray      # shape (T,) seconds
    uav_id: int
    reached_target: bool
    n_collisions: int = 0
    algorithm: str = "unknown"

    def __post_init__(self):
        self.waypoints = np.asarray(self.waypoints, dtype=float)
        self.velocities = np.asarray(self.velocities, dtype=float)
        self.accelerations = np.asarray(self.accelerations, dtype=float)
        self.timestamps = np.asarray(self.timestamps, dtype=float)

    @property
    def duration(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def n_steps(self) -> int:
        return len(self.waypoints)


class ObjectiveFunction(ABC):
    """Abstract base for all objective functions."""
    name: str = "base"
    minimize: bool = True  # True = lower is better

    @abstractmethod
    def evaluate(self, path: PathResult, obstacle_map: "ObstacleMap") -> float:
        ...

    def normalized(self, value: float, population: List[float]) -> float:
        """Normalize value to [0, 1] using population min-max.

        Returns 0.0 if this objective's direction says this value is worst,
        1.0 if it's best, accounting for minimize/maximize convention.
        """
        if len(population) < 2:
            return 0.5
        lo, hi = min(population), max(population)
        if abs(hi - lo) < 1e-10:
            return 0.5
        n = (value - lo) / (hi - lo)
        return float(n if self.minimize else 1.0 - n)
