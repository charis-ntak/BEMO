"""Tests for objective functions."""
import numpy as np
import pytest

from bemo.objectives import (PathLengthObjective, EnergyObjective,
                               SmoothnessObjective, SafetyObjective)
from bemo.objectives.base import PathResult
from tests.fixtures import make_straight_path, make_obstacle_map, BOUNDS


@pytest.fixture
def straight_path():
    return make_straight_path(start=(0, 0, 1), end=(8, 0, 1), n=100)


@pytest.fixture
def omap():
    return make_obstacle_map()


class TestPathLengthObjective:
    def test_straight_path(self, straight_path, omap):
        obj = PathLengthObjective()
        length = obj.evaluate(straight_path, omap)
        assert abs(length - 8.0) < 0.2  # ~8m from (0,0,1) to (8,0,1)

    def test_single_point(self, omap):
        obj = PathLengthObjective()
        path = PathResult(
            waypoints=np.array([[0, 0, 0]]),
            velocities=np.zeros((1, 3)),
            accelerations=np.zeros((1, 3)),
            timestamps=np.array([0.0]),
            uav_id=0, reached_target=False,
        )
        assert obj.evaluate(path, omap) == 0.0

    def test_minimize_flag(self):
        assert PathLengthObjective().minimize is True

    def test_normalized(self):
        obj = PathLengthObjective()
        pop = [5.0, 10.0, 15.0]
        # minimize=True: 5.0 is best (lowest), normalized to 0; 15.0 is worst, to 1
        assert obj.normalized(5.0, pop) == pytest.approx(0.0)
        assert obj.normalized(15.0, pop) == pytest.approx(1.0)


class TestEnergyObjective:
    def test_zero_acceleration(self, omap):
        obj = EnergyObjective()
        path = PathResult(
            waypoints=np.zeros((10, 3)),
            velocities=np.zeros((10, 3)),
            accelerations=np.zeros((10, 3)),
            timestamps=np.linspace(0, 1, 10),
            uav_id=0, reached_target=False,
        )
        assert obj.evaluate(path, omap) == pytest.approx(0.0)

    def test_positive_with_acceleration(self, omap):
        obj = EnergyObjective()
        path = PathResult(
            waypoints=np.zeros((10, 3)),
            velocities=np.zeros((10, 3)),
            accelerations=np.ones((10, 3)) * 2.0,
            timestamps=np.linspace(0, 1, 10),
            uav_id=0, reached_target=False,
        )
        val = obj.evaluate(path, omap)
        assert val > 0

    def test_minimize_flag(self):
        assert EnergyObjective().minimize is True


class TestSmoothnessObjective:
    def test_straight_line(self, omap):
        """Straight line: velocity and acceleration parallel → zero curvature."""
        obj = SmoothnessObjective()
        n = 50
        vels = np.tile([1, 0, 0], (n, 1)).astype(float)
        accs = np.zeros((n, 3))
        path = PathResult(
            waypoints=np.column_stack([np.linspace(0, 5, n), np.zeros(n), np.zeros(n)]),
            velocities=vels,
            accelerations=accs,
            timestamps=np.linspace(0, 5, n),
            uav_id=0, reached_target=True,
        )
        val = obj.evaluate(path, omap)
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_minimize_flag(self):
        assert SmoothnessObjective().minimize is True


class TestSafetyObjective:
    def test_free_path(self, omap):
        """Path in open space should have positive clearance."""
        obj = SafetyObjective()
        path = make_straight_path(start=(0, 0, 0), end=(2, 0, 0), n=20)
        val = obj.evaluate(path, omap)
        assert isinstance(val, float)

    def test_maximize_flag(self):
        assert SafetyObjective().minimize is False

    def test_normalized_safety(self):
        obj = SafetyObjective()
        # minimize=False: normalized() returns a cost (0=best, 1=worst).
        # For safety: highest clearance is best → cost 0.0; lowest → cost 1.0.
        pop = [0.5, 1.0, 2.0]
        # 2.0 is best clearance → cost = 0.0
        assert obj.normalized(2.0, pop) == pytest.approx(0.0)
        # 0.5 is worst clearance → cost = 1.0
        assert obj.normalized(0.5, pop) == pytest.approx(1.0)

    def test_no_obstacles(self):
        from bemo.environment.obstacles import ObstacleMap
        omap = ObstacleMap([], BOUNDS)
        obj = SafetyObjective()
        path = make_straight_path()
        val = obj.evaluate(path, omap)
        assert val == float("inf")
