"""PPO planner using stable-baselines3."""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from bemo.objectives.base import PathResult
from .base import PlannerBase, PlannerConfig

if TYPE_CHECKING:
    from bemo.environment.obstacles import ObstacleMap
    from bemo.environment.uav import UAVDynamics

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class PPOPlanner(PlannerBase):
    """PPO-based planner.

    Training: call .train() to train the policy on randomly generated scenarios.
    Planning: run the trained policy deterministically to collect a trajectory.

    Multi-agent support: parameter sharing — all UAVs share one policy,
    distinguished by their observation (which includes uav_id).
    """

    def __init__(self,
                 config: Optional[PlannerConfig] = None,
                 env_config: Optional[dict] = None,
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 net_arch: Optional[List[int]] = None,
                 model_path: Optional[str] = None):
        config = config or PlannerConfig()
        super().__init__(config)
        self.env_config = env_config or {}
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.net_arch = net_arch or [256, 256]
        self._model = None

        if model_path:
            self.load(model_path)

    def name(self) -> str:
        return "PPO"

    def train(self,
              total_timesteps: int = 1_000_000,
              n_envs: int = 4,
              log_dir: str = "logs/ppo",
              save_path: str = "models/ppo_uav") -> None:
        """Train PPO on vectorized UAVNavEnv."""
        if not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 not installed. Run: pip install stable-baselines3")

        from bemo.environment.gym_env import UAVNavEnv
        import os
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        def make_env(rank: int):
            def _init():
                env = UAVNavEnv(self.env_config, uav_id=0)
                env.reset(seed=rank * 1000)
                return env
            return _init

        # Use DummyVecEnv for small n_envs (avoids multiprocessing overhead)
        if n_envs <= 2:
            vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
        else:
            try:
                vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
            except Exception:
                vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

        vec_env = VecMonitor(vec_env, log_dir)

        self._model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs={"net_arch": self.net_arch},
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            verbose=1,
            tensorboard_log=log_dir,
            seed=self.config.seed,
        )
        self._model.learn(total_timesteps=total_timesteps)
        self._model.save(save_path)
        vec_env.close()
        print(f"PPO model saved to {save_path}")

    def plan(self,
             start: np.ndarray,
             target: np.ndarray,
             obstacle_map: "ObstacleMap",
             bounds: np.ndarray,
             dynamics: "UAVDynamics") -> PathResult:
        """Run trained policy to collect a trajectory."""
        if self._model is None:
            raise RuntimeError("PPOPlanner must be trained or loaded before plan(). "
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
        obs, _ = env.reset.__wrapped__(env) if hasattr(env.reset, '__wrapped__') else (env.world.get_observation(0), {})

        # Manually set up the environment state
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
        self._model = PPO.load(path)

    @classmethod
    def from_config(cls, config: dict, planner_cfg: Optional[PlannerConfig] = None) -> "PPOPlanner":
        ppo_cfg = config.get("ppo", {})
        return cls(
            config=planner_cfg,
            env_config=config,
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            n_steps=ppo_cfg.get("n_steps", 2048),
            batch_size=ppo_cfg.get("batch_size", 64),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            net_arch=ppo_cfg.get("net_arch", [256, 256]),
        )
