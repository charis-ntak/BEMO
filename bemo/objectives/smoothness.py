"""Smoothness objective: minimize mean path curvature."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ObjectiveFunction, PathResult

if TYPE_CHECKING:
    from bemo.environment.obstacles import ObstacleMap


class SmoothnessObjective(ObjectiveFunction):
    """Minimize mean curvature κ = ||v × a|| / ||v||³."""
    name = "smoothness"
    minimize = True

    def evaluate(self, path: PathResult, obstacle_map: "ObstacleMap") -> float:
        if path.n_steps < 2:
            return 0.0
        v = path.velocities
        a = path.accelerations
        cross = np.cross(v, a)
        cross_norm = np.linalg.norm(cross, axis=1)
        v_norm = np.linalg.norm(v, axis=1)
        v_norm_cubed = v_norm ** 3
        mask = v_norm_cubed > 1e-6
        kappas = np.where(mask, cross_norm / np.where(mask, v_norm_cubed, 1.0), 0.0)
        return float(np.mean(kappas))
