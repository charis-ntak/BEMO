"""Tests for RRT* planner."""
import numpy as np
import pytest

from bemo.algorithms.base import PlannerConfig
from bemo.algorithms.rrt_star import RRTStarPlanner
from bemo.environment.uav import UAVDynamics
from tests.fixtures import BOUNDS, make_obstacle_map, make_box, make_cylinder


@pytest.fixture
def dynamics():
    return UAVDynamics(dt=0.1)


@pytest.fixture
def simple_omap():
    # Simple obstacle map with one obstacle not in the path
    return make_obstacle_map([make_box(pos=(5, 5, 2), half=(0.5, 0.5, 0.5))])


@pytest.fixture
def planner():
    cfg = PlannerConfig(max_iterations=500, timeout_seconds=10.0, seed=42)
    return RRTStarPlanner(config=cfg, step_size=0.5, neighbor_radius=1.0, goal_bias=0.15)


class TestRRTStar:
    def test_plan_returns_path_result(self, planner, simple_omap, dynamics):
        from bemo.objectives.base import PathResult
        start = np.array([1.0, 1.0, 1.0])
        target = np.array([8.0, 8.0, 1.0])
        path = planner.plan(start, target, simple_omap, BOUNDS, dynamics)
        assert isinstance(path, PathResult)

    def test_path_starts_at_start(self, planner, simple_omap, dynamics):
        start = np.array([1.0, 1.0, 1.0])
        target = np.array([8.0, 8.0, 1.0])
        path = planner.plan(start, target, simple_omap, BOUNDS, dynamics)
        assert path.n_steps > 0
        assert np.linalg.norm(path.waypoints[0] - start) < 1.0

    def test_path_shapes_consistent(self, planner, simple_omap, dynamics):
        start = np.array([1.0, 1.0, 1.0])
        target = np.array([8.0, 8.0, 1.0])
        path = planner.plan(start, target, simple_omap, BOUNDS, dynamics)
        n = path.n_steps
        assert path.waypoints.shape == (n, 3)
        assert path.velocities.shape == (n, 3)
        assert path.accelerations.shape == (n, 3)
        assert path.timestamps.shape == (n,)

    def test_name(self, planner):
        assert planner.name() == "RRT*"

    def test_open_space_reaches_target(self, dynamics):
        """In open space, RRT* should reach target."""
        from bemo.environment.obstacles import ObstacleMap
        omap = ObstacleMap([], BOUNDS)
        cfg = PlannerConfig(max_iterations=2000, timeout_seconds=30.0, seed=0)
        p = RRTStarPlanner(config=cfg, step_size=0.4, goal_bias=0.15)
        start = np.array([0.5, 0.5, 0.5])
        target = np.array([9.0, 9.0, 4.0])
        path = p.plan(start, target, omap, BOUNDS, dynamics)
        # With enough iterations and open space, should reach or get close
        dist = np.linalg.norm(path.waypoints[-1] - target)
        assert dist < 3.0 or path.reached_target

    def test_edge_cost_length_only(self, planner, simple_omap):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 0.0, 0.0])
        cost = planner._edge_cost(p1, p2, None)
        # length weight = 1.0, no safety weight without obstacle map
        assert abs(cost - 3.0) < 0.01
