"""Safety objective: maximize minimum obstacle clearance along path."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ObjectiveFunction, PathResult

if TYPE_CHECKING:
    from bemo.environment.obstacles import ObstacleMap


class SafetyObjective(ObjectiveFunction):
    """Maximize minimum SDF clearance from obstacles."""
    name = "safety"
    minimize = False  # higher clearance = better

    def evaluate(self, path: PathResult, obstacle_map: "ObstacleMap") -> float:
        if path.n_steps == 0 or obstacle_map is None:
            return 0.0
        clearances = [obstacle_map.clearance(pt) for pt in path.waypoints]
        return float(np.min(clearances))
