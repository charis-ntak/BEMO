"""Gymnasium-compatible environment wrappers for single and multi-agent UAV navigation."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .obstacles import ObstacleMap
from .scenario_generator import Scenario, ScenarioGenerator
from .uav import UAVState
from .world import SimulationWorld


class UAVNavEnv(gym.Env):
    """Single-agent Gymnasium environment for UAV navigation.

    Observation space: [pos(3), vel(3), acc(3), heading(1), uav_id(1),
                        target_rel(3), K×(sdf,dx,dy,dz)]
    Action space: 3D acceleration command.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict, uav_id: int = 0,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.config = config
        self.uav_id = uav_id
        self.render_mode = render_mode

        # Build world
        self.world = SimulationWorld(config)

        env_cfg = config.get("environment", {})
        bounds_raw = env_cfg.get("bounds", [[0, 0, 0], [10, 10, 5]])
        self.bounds = np.array(bounds_raw, dtype=float)

        scenario_cfg = config.get("scenario", {})
        self._gen = ScenarioGenerator(self.bounds,
                                       seed=scenario_cfg.get("seed", None))
        self._n_obstacles = scenario_cfg.get("n_obstacles", 5)
        self._min_clearance = scenario_cfg.get("min_clearance", 0.5)

        gym_cfg = config.get("gym", {})
        K = int(gym_cfg.get("obs_obstacle_count", 5))
        obs_dim = 11 + 3 + K * 4  # pos(3)+vel(3)+acc(3)+heading(1)+uav_id(1) + target_rel(3) + K*(sdf,dx,dy,dz)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        uav_cfg = config.get("uav", {})
        max_acc = float(uav_cfg.get("max_acceleration", 3.0))
        self.action_space = spaces.Box(
            low=-max_acc, high=max_acc, shape=(3,), dtype=np.float32
        )

        self._max_steps = int(gym_cfg.get("max_episode_steps", 500))
        self._reward_weights = gym_cfg.get("reward_weights", {})
        self._step_count = 0
        self._last_reached = False
        self._collision_count = 0
        self._prev_dist: Optional[float] = None

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._gen = ScenarioGenerator(self.bounds, seed=seed)

        scenario = self._gen.generate(
            n_obstacles=self._n_obstacles,
            n_uavs=1,
            min_clearance=self._min_clearance,
        )
        self.world.reset(scenario)
        self._step_count = 0
        self._last_reached = False
        self._collision_count = 0
        target = self.world.targets[self.uav_id]
        pos = self.world.states[self.uav_id].position
        self._prev_dist = float(np.linalg.norm(target - pos))

        obs = self.world.get_observation(self.uav_id)
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=float)
        results = self.world.step({self.uav_id: action})
        new_state, reached, collision = results[self.uav_id]

        self._step_count += 1
        self._last_reached = reached
        if collision:
            self._collision_count += 1

        reward = self._compute_reward(new_state, reached, collision, action)
        terminated = reached or collision
        truncated = self._step_count >= self._max_steps

        obs = self.world.get_observation(self.uav_id)
        info = {
            "reached": reached,
            "collision": collision,
            "step": self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, state: UAVState,
                         reached: bool, collision: bool,
                         action: np.ndarray) -> float:
        w = self._reward_weights
        target = self.world.targets[self.uav_id]
        dist = float(np.linalg.norm(target - state.position))

        # Progress reward (change in distance to target)
        progress = 0.0
        if self._prev_dist is not None:
            progress = (self._prev_dist - dist) * w.get("progress", 1.0)
        self._prev_dist = dist

        # Clearance bonus
        clearance = self.world.obstacle_map.clearance(state.position)
        clearance_bonus = min(clearance - self.world.uav_radius, 0.5) * w.get("clearance_bonus", 0.1)

        # Reach bonus
        reach_bonus = w.get("reach_bonus", 100.0) if reached else 0.0

        # Collision penalty
        collision_pen = w.get("collision_penalty", -50.0) if collision else 0.0

        # Jerk penalty (smoothness)
        jerk_pen = -float(np.linalg.norm(action - state.acceleration)) * abs(w.get("jerk_penalty", 0.01))

        return float(progress + clearance_bonus + reach_bonus + collision_pen + jerk_pen)

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            print(f"Step {self._step_count}: UAV at "
                  f"{self.world.states[self.uav_id].position.round(2)}")
        return None

    def close(self):
        pass


class MultiUAVEnv(gym.Env):
    """Multi-agent wrapper: each agent shares one policy (parameter sharing).

    Observation and action spaces are per-agent (identical structure).
    Agents are indexed 0..n_uavs-1.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        scenario_cfg = config.get("scenario", {})
        self.n_uavs = int(scenario_cfg.get("n_uavs", 2))

        # Use single-agent env to get obs/action spaces
        self._ref_env = UAVNavEnv(config, uav_id=0)
        self.observation_space = self._ref_env.observation_space
        self.action_space = self._ref_env.action_space

        # Shared world
        self.world = SimulationWorld(config)
        env_cfg = config.get("environment", {})
        bounds_raw = env_cfg.get("bounds", [[0, 0, 0], [10, 10, 5]])
        self.bounds = np.array(bounds_raw, dtype=float)
        self._gen = ScenarioGenerator(self.bounds, seed=scenario_cfg.get("seed", None))
        self._n_obstacles = scenario_cfg.get("n_obstacles", 5)
        self._min_clearance = scenario_cfg.get("min_clearance", 0.5)
        self._max_steps = int(config.get("gym", {}).get("max_episode_steps", 500))
        self._step_count = 0
        self._prev_dists: Dict[int, float] = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        scenario = self._gen.generate(
            n_obstacles=self._n_obstacles,
            n_uavs=self.n_uavs,
            min_clearance=self._min_clearance,
        )
        self.world.reset(scenario)
        self._step_count = 0
        self._prev_dists = {}
        for i in range(self.n_uavs):
            pos = self.world.states[i].position
            tgt = self.world.targets[i]
            self._prev_dists[i] = float(np.linalg.norm(tgt - pos))

        obs_list = [self.world.get_observation(i) for i in range(self.n_uavs)]
        return obs_list, {}

    def step(self, actions: List[np.ndarray]):
        actions_dict = {i: np.asarray(a, dtype=float) for i, a in enumerate(actions)}
        results = self.world.step(actions_dict)
        self._step_count += 1

        obs_list, rewards, terminateds, truncateds, infos = [], [], [], [], []
        all_done = True
        for i in range(self.n_uavs):
            new_state, reached, collision = results[i]
            target = self.world.targets[i]
            dist = float(np.linalg.norm(target - new_state.position))
            progress = (self._prev_dists[i] - dist)
            self._prev_dists[i] = dist

            reward = progress - (50.0 if collision else 0.0) + (100.0 if reached else 0.0)
            terminated = reached or collision
            truncated = self._step_count >= self._max_steps

            obs_list.append(self.world.get_observation(i))
            rewards.append(float(reward))
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append({"reached": reached, "collision": collision})
            if not (terminated or truncated):
                all_done = False

        return obs_list, rewards, terminateds, truncateds, infos
