"""Path metrics computation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bemo.objectives.base import ObjectiveFunction, PathResult
    from bemo.environment.obstacles import ObstacleMap


@dataclass
class PathMetrics:
    """All metrics for one algorithm on one scenario run."""
    algorithm: str
    uav_id: int
    scenario_id: str

    # Raw objective values
    path_length_m: float = 0.0
    energy_proxy: float = 0.0
    mean_curvature: float = 0.0
    min_clearance_m: float = 0.0

    # Derived metrics
    reached_target: bool = False
    n_collisions: int = 0
    planning_time_s: float = 0.0
    path_duration_s: float = 0.0
    n_waypoints: int = 0

    # Aggregated scores (filled in after all algorithms run)
    pareto_rank: Optional[int] = None
    fuzzy_fitness: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "uav_id": self.uav_id,
            "scenario_id": self.scenario_id,
            "path_length_m": self.path_length_m,
            "energy_proxy": self.energy_proxy,
            "mean_curvature": self.mean_curvature,
            "min_clearance_m": self.min_clearance_m,
            "reached_target": self.reached_target,
            "n_collisions": self.n_collisions,
            "planning_time_s": self.planning_time_s,
            "path_duration_s": self.path_duration_s,
            "n_waypoints": self.n_waypoints,
            "pareto_rank": self.pareto_rank,
            "fuzzy_fitness": self.fuzzy_fitness,
        }


def compute_metrics(path: "PathResult",
                     obstacle_map: "ObstacleMap",
                     objectives: List["ObjectiveFunction"],
                     planning_time: float,
                     scenario_id: str = "unknown") -> PathMetrics:
    """Compute all metrics for a path."""
    obj_map = {fn.name: fn for fn in objectives}

    length = obj_map["path_length"].evaluate(path, obstacle_map) if "path_length" in obj_map else 0.0
    energy = obj_map["energy"].evaluate(path, obstacle_map) if "energy" in obj_map else 0.0
    smooth = obj_map["smoothness"].evaluate(path, obstacle_map) if "smoothness" in obj_map else 0.0
    safety = obj_map["safety"].evaluate(path, obstacle_map) if "safety" in obj_map else 0.0

    return PathMetrics(
        algorithm=path.algorithm,
        uav_id=path.uav_id,
        scenario_id=scenario_id,
        path_length_m=length,
        energy_proxy=energy,
        mean_curvature=smooth,
        min_clearance_m=safety,
        reached_target=path.reached_target,
        n_collisions=path.n_collisions,
        planning_time_s=planning_time,
        path_duration_s=path.duration,
        n_waypoints=path.n_steps,
    )
