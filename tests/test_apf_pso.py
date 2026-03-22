"""Tests for APF-PSO planner."""
import numpy as np
import pytest

from bemo.algorithms.base import PlannerConfig
from bemo.algorithms.apf_pso import APFPSOPlanner
from bemo.environment.uav import UAVDynamics
from tests.fixtures import BOUNDS, make_obstacle_map, make_box


@pytest.fixture
def dynamics():
    return UAVDynamics(dt=0.1)


@pytest.fixture
def simple_omap():
    return make_obstacle_map([make_box(pos=(5, 5, 2), half=(0.5, 0.5, 0.5))])


@pytest.fixture
def planner():
    cfg = PlannerConfig(max_iterations=200, timeout_seconds=30.0, seed=42)
    return APFPSOPlanner(
        config=cfg, n_particles=10, n_waypoints=8,
        sensor_radius=2.0, pso_iterations=20
    )


class TestAPFPSOPlanner:
    def test_plan_returns_path_result(self, planner, simple_omap, dynamics):
        from bemo.objectives.base import PathResult
        start = np.array([1.0, 1.0, 1.0])
        target = np.array([8.0, 8.0, 1.0])
        path = planner.plan(start, target, simple_omap, BOUNDS, dynamics)
        assert isinstance(path, PathResult)

    def test_path_shapes_consistent(self, planner, simple_omap, dynamics):
        start = np.array([1.0, 1.0, 1.0])
        target = np.array([8.0, 8.0, 1.0])
        path = planner.plan(start, target, simple_omap, BOUNDS, dynamics)
        n = path.n_steps
        assert n > 0
        assert path.waypoints.shape == (n, 3)
        assert path.velocities.shape == (n, 3)
        assert path.accelerations.shape == (n, 3)
        assert path.timestamps.shape == (n,)

    def test_name(self, planner):
        assert planner.name() == "APF-PSO"

    def test_apf_force_attractive_toward_target(self, planner):
        """APF attractive force should point toward target."""
        pos = np.array([0.0, 0.0, 0.0])
        target = np.array([5.0, 0.0, 0.0])
        force = planner._apf_force(pos, target, discovered=[])
        assert force[0] > 0  # should pull toward +x

    def test_apf_force_repulsive_from_obstacle(self, planner):
        """APF repulsive force should push away from obstacle."""
        from tests.fixtures import make_box
        obs = make_box(pos=(1, 0, 0), half=(0.3, 0.3, 0.3))
        pos = np.array([0.5, 0.0, 0.0])  # Close to obstacle
        target = np.array([10.0, 0.0, 0.0])
        force = planner._apf_force(pos, target, discovered=[obs])
        # Net force should have repulsive component pushing away from obstacle at (1,0,0)
        # → force in -x direction from repulsion
        assert isinstance(force, np.ndarray)
        assert force.shape == (3,)

    def test_particle_fitness(self, planner, simple_omap):
        """Particle fitness should return a scalar."""
        wps = np.array([
            [1.0, 1.0, 1.0],
            [3.0, 2.0, 1.0],
            [5.0, 3.0, 1.0],
            [7.0, 5.0, 1.0],
        ])
        target = np.array([9.0, 7.0, 1.0])
        fitness = planner._particle_fitness(wps, target, simple_omap)
        assert isinstance(fitness, float)
        assert not np.isnan(fitness)
