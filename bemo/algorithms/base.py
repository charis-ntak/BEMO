"""Abstract base class for all path planners."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bemo.environment.obstacles import ObstacleMap
    from bemo.environment.uav import UAVDynamics
    from bemo.objectives.base import PathResult


@dataclass
class PlannerConfig:
    max_iterations: int = 5000
    timeout_seconds: float = 30.0
    uav_radius: float = 0.15
    seed: Optional[int] = None


class PlannerBase(ABC):
    """Interface all planners must implement for uniform benchmarking."""

    def __init__(self, config: PlannerConfig):
        self.config = config

    @abstractmethod
    def plan(self,
             start: np.ndarray,
             target: np.ndarray,
             obstacle_map: "ObstacleMap",
             bounds: np.ndarray,
             dynamics: "UAVDynamics") -> "PathResult":
        """Plan a path from start to target.

        Returns a PathResult regardless of whether target is reached.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
