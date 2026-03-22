"""DDPG planner using stable-baselines3."""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from bemo.objectives.base import PathResult
from .base import PlannerBase, PlannerConfig

if TYPE_CHECKING:
    from bemo.environment.obstacles import ObstacleMap
    from bemo.environment.uav import UAVDynamics

try:
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class DDPGPlanner(PlannerBase):
    """DDPG-based planner.

    Uses Ornstein-Uhlenbeck action noise for temporally-correlated exploration,
    which is well-suited to UAV dynamics where smooth exploration is needed.
    Off-policy learning with replay buffer for sample efficiency.
    """

    def __init__(self,
                 config: Optional[PlannerConfig] = None,
                 env_config: Optional[dict] = None,
                 learning_rate: float = 1e-3,
                 buffer_size: int = 1_000_000,
                 batch_size: int = 256,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 train_freq: int = 1,
                 action_noise_sigma: float = 0.1,
                 net_arch: Optional[List[int]] = None,
                 model_path: Optional[str] = None):
        config = config or PlannerConfig()
        super().__init__(config)
        self.env_config = env_config or {}
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.action_noise_sigma = action_noise_sigma
        self.net_arch = net_arch or [400, 300]
        self._model = None

        if model_path:
            self.load(model_path)

    def name(self) -> str:
        return "DDPG"

    def train(self,
              total_timesteps: int = 500_000,
              log_dir: str = "logs/ddpg",
              save_path: str = "models/ddpg_uav") -> None:
        """Train DDPG on a single UAVNavEnv."""
        if not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 not installed. Run: pip install stable-baselines3")

        from bemo.environment.gym_env import UAVNavEnv
        import os
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        env = UAVNavEnv(self.env_config, uav_id=0)
        n_actions = env.action_space.shape[0]

        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=self.action_noise_sigma * np.ones(n_actions),
        )

        self._model = DDPG(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            policy_kwargs={"net_arch": self.net_arch},
            verbose=1,
            tensorboard_log=log_dir,
            seed=self.config.seed,
        )
        self._model.learn(total_timesteps=total_timesteps)
        self._model.save(save_path)
        env.close()
        print(f"DDPG model saved to {save_path}")

    def plan(self,
             start: np.ndarray,
             target: np.ndarray,
             obstacle_map: "ObstacleMap",
             bounds: np.ndarray,
             dynamics: "UAVDynamics") -> PathResult:
        """Run trained policy to collect a trajectory."""
        if self._model is None:
            raise RuntimeError("DDPGPlanner must be trained or loaded before plan(). "
                               "Call .train() or .load(path) first.")

        from bemo.environment.gym_env import UAVNavEnv
        from bemo.environment.scenario_generator import Scenario

        scenario = Scenario(
            obstacles=obstacle_map.obstacles,
            starts=[start.copy()],
            targets=[target.copy()],
            bounds=bounds,
            n_uavs=1,
            seed=0,
        )
        env = UAVNavEnv(self.env_config, uav_id=0)
        env.world.reset(scenario)
        env._step_count = 0
        env._last_reached = False
        env._collision_count = 0
        tgt = env.world.targets[0]
        pos = env.world.states[0].position
        env._prev_dist = float(np.linalg.norm(tgt - pos))
        obs = env.world.get_observation(0)

        positions = [env.world.states[0].position.copy()]
        velocities = [env.world.states[0].velocity.copy()]
        accelerations = [env.world.states[0].acceleration.copy()]
        timestamps = [0.0]
        t = 0.0
        done = False
        max_steps = int(self.env_config.get("gym", {}).get("max_episode_steps", 500))

        while not done and len(timestamps) < max_steps:
            action, _ = self._model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            state = env.world.states[0]
            t += dynamics.dt
            positions.append(state.position.copy())
            velocities.append(state.velocity.copy())
            accelerations.append(state.acceleration.copy())
            timestamps.append(t)
            done = terminated or truncated

        return PathResult(
            waypoints=np.array(positions),
            velocities=np.array(velocities),
            accelerations=np.array(accelerations),
            timestamps=np.array(timestamps),
            uav_id=0,
            reached_target=env._last_reached,
            n_collisions=env._collision_count,
            algorithm=self.name(),
        )

    def save(self, path: str) -> None:
        if self._model is not None:
            self._model.save(path)

    def load(self, path: str) -> None:
        if not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 not installed.")
        self._model = DDPG.load(path)

    @classmethod
    def from_config(cls, config: dict, planner_cfg: Optional[PlannerConfig] = None) -> "DDPGPlanner":
        ddpg_cfg = config.get("ddpg", {})
        return cls(
            config=planner_cfg,
            env_config=config,
            learning_rate=ddpg_cfg.get("learning_rate", 1e-3),
            buffer_size=ddpg_cfg.get("buffer_size", 1_000_000),
            batch_size=ddpg_cfg.get("batch_size", 256),
            tau=ddpg_cfg.get("tau", 0.005),
            gamma=ddpg_cfg.get("gamma", 0.99),
            action_noise_sigma=ddpg_cfg.get("action_noise_sigma", 0.1),
            net_arch=ddpg_cfg.get("net_arch", [400, 300]),
        )
