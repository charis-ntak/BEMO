"""Energy objective: minimize integral of squared acceleration (thrust energy proxy)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ObjectiveFunction, PathResult

if TYPE_CHECKING:
    from bemo.environment.obstacles import ObstacleMap


class EnergyObjective(ObjectiveFunction):
    name = "energy"
    minimize = True

    def evaluate(self, path: PathResult, obstacle_map: "ObstacleMap") -> float:
        if path.n_steps < 2:
            return 0.0
        acc_sq = np.sum(path.accelerations ** 2, axis=1)
        try:
            return float(np.trapezoid(acc_sq, path.timestamps))
        except AttributeError:
            return float(np.trapz(acc_sq, path.timestamps))
