"""Path length objective: minimize total Euclidean distance traveled."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ObjectiveFunction, PathResult

if TYPE_CHECKING:
    from bemo.environment.obstacles import ObstacleMap


class PathLengthObjective(ObjectiveFunction):
    name = "path_length"
    minimize = True

    def evaluate(self, path: PathResult, obstacle_map: "ObstacleMap") -> float:
        if path.n_steps < 2:
            return 0.0
        diffs = np.diff(path.waypoints, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
