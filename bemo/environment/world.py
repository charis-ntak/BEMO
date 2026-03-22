"""Simulation world: top-level container for UAVs and obstacles."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .obstacles import ObstacleMap
from .scenario_generator import Scenario
from .uav import UAVDynamics, UAVState


class SimulationWorld:
    """Owns the obstacle map and all UAVs. Advances simulation time."""

    def __init__(self, config: dict):
        env_cfg = config.get("environment", {})
        bounds_raw = env_cfg.get("bounds", [[0, 0, 0], [10, 10, 5]])
        self.bounds = np.array(bounds_raw, dtype=float)
        self.dt = float(env_cfg.get("dt", 0.05))

        self.dynamics = UAVDynamics.from_config(config)
        self.uav_radius = float(config.get("uav", {}).get("radius", 0.15))
        self.obs_obstacle_count = int(config.get("gym", {}).get("obs_obstacle_count", 5))

        self.obstacle_map: Optional[ObstacleMap] = None
        self.states: Dict[int, UAVState] = {}
        self.targets: Dict[int, np.ndarray] = {}
        self._scenario: Optional[Scenario] = None

    def reset(self, scenario: Scenario) -> None:
        """Load a scenario: build obstacle map and initialize UAVs."""
        self._scenario = scenario
        self.obstacle_map = ObstacleMap(scenario.obstacles, self.bounds)
        self.states = {}
        self.targets = {}

        for i in range(scenario.n_uavs):
            self.states[i] = UAVState(
                position=scenario.starts[i].copy(),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                heading=0.0,
                uav_id=i,
            )
            self.targets[i] = scenario.targets[i].copy()

    def step(self, actions: Dict[int, np.ndarray]) -> Dict[int, Tuple[UAVState, bool, bool]]:
        """Advance simulation one timestep.

        Args:
            actions: dict mapping uav_id -> acceleration command (3,).

        Returns:
            dict mapping uav_id -> (new_state, reached_target, collision).
        """
        results = {}
        for uav_id, action in actions.items():
            state = self.states[uav_id]
            new_state = self.dynamics.step(state, action)

            # Clip to bounds
            new_state.position = np.clip(
                new_state.position, self.bounds[0], self.bounds[1]
            )

            collision = not self.obstacle_map.is_collision_free(
                new_state.position, self.uav_radius
            )
            reached = (np.linalg.norm(new_state.position - self.targets[uav_id])
                       < self.uav_radius * 2)

            self.states[uav_id] = new_state
            results[uav_id] = (new_state, reached, collision)

        return results

    def get_observation(self, uav_id: int, obs_radius: float = 3.0) -> np.ndarray:
        """Build observation vector for one UAV.

        Returns: [pos(3), vel(3), acc(3), heading(1), uav_id(1),
                  target_rel(3), K×(sdf,dx,dy,dz)] shape = 10+3+K*4
        """
        state = self.states[uav_id]
        target = self.targets[uav_id]

        own_vec = state.to_vector()  # 10-dim
        target_rel = (target - state.position).astype(np.float32)  # 3-dim

        # K nearest obstacle observations
        K = self.obs_obstacle_count
        nearest = self.obstacle_map.nearest_obstacles(state.position, k=K)
        obs_feats = []
        for obs in nearest:
            sdf_val = obs.sdf(state.position)
            direction = obs.position - state.position
            obs_feats.extend([sdf_val, direction[0], direction[1], direction[2]])
        # Pad if fewer than K obstacles
        while len(obs_feats) < K * 4:
            obs_feats.extend([100.0, 0.0, 0.0, 0.0])

        return np.concatenate([own_vec, target_rel, obs_feats[:K * 4]]).astype(np.float32)

    @property
    def n_uavs(self) -> int:
        return len(self.states)
