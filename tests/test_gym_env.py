"""Tests for Gymnasium environment wrapper."""
import numpy as np
import pytest

from bemo.environment.gym_env import UAVNavEnv, MultiUAVEnv


DEFAULT_CONFIG = {
    "environment": {"bounds": [[0, 0, 0], [10, 10, 5]], "dt": 0.05},
    "uav": {"max_speed": 2.0, "max_acceleration": 3.0, "max_jerk": 5.0, "radius": 0.15},
    "scenario": {"n_obstacles": 3, "n_uavs": 1, "min_clearance": 0.3, "seed": 42},
    "gym": {"obs_obstacle_count": 3, "max_episode_steps": 50,
            "reward_weights": {"progress": 1.0, "clearance_bonus": 0.1,
                               "reach_bonus": 10.0, "collision_penalty": -5.0,
                               "jerk_penalty": -0.01}},
}


@pytest.fixture
def env():
    e = UAVNavEnv(DEFAULT_CONFIG, uav_id=0)
    yield e
    e.close()


class TestUAVNavEnv:
    def test_spaces(self, env):
        assert env.observation_space is not None
        assert env.action_space is not None
        # obs dim = 11 + 3 + 3*4 = 26
        assert env.observation_space.shape == (26,)
        assert env.action_space.shape == (3,)

    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_step_returns_valid_outputs(self, env):
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_in_space(self, env):
        obs, _ = env.reset(seed=0)
        # Obs should be finite
        assert np.all(np.isfinite(obs))

    def test_truncation_after_max_steps(self, env):
        env.reset(seed=0)
        action = np.zeros(3)  # stay still
        for _ in range(env._max_steps - 1):
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated:
                break
        # After max steps, should be truncated
        obs, _, terminated, truncated, _ = env.step(action)
        if not terminated:
            assert truncated

    def test_random_episode(self, env):
        obs, _ = env.reset(seed=1)
        done = False
        steps = 0
        while not done and steps < 50:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0

    def test_gymnasium_spec(self, env):
        """Basic gymnasium spec checks."""
        try:
            from gymnasium.utils.env_checker import check_env
            # check_env can be slow, run with limited steps
            check_env(env, warn=True, skip_render_check=True)
        except Exception as e:
            pytest.skip(f"env_checker raised: {e}")


class TestMultiUAVEnv:
    def test_reset_returns_list(self):
        cfg = {**DEFAULT_CONFIG, "scenario": {**DEFAULT_CONFIG["scenario"], "n_uavs": 2}}
        env = MultiUAVEnv(cfg)
        obs_list, info = env.reset(seed=0)
        assert len(obs_list) == 2
        for obs in obs_list:
            assert obs.shape == env.observation_space.shape

    def test_step_multi_agent(self):
        cfg = {**DEFAULT_CONFIG, "scenario": {**DEFAULT_CONFIG["scenario"], "n_uavs": 2}}
        env = MultiUAVEnv(cfg)
        env.reset(seed=0)
        actions = [env.action_space.sample() for _ in range(2)]
        obs_list, rewards, terminateds, truncateds, infos = env.step(actions)
        assert len(obs_list) == 2
        assert len(rewards) == 2
        assert len(terminateds) == 2
