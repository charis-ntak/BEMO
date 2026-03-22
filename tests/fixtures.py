"""Shared test fixtures."""
import numpy as np
import pytest

from bemo.environment.obstacles import BoxObstacle, CylinderObstacle, ObstacleMap
from bemo.environment.uav import UAVState, UAVDynamics
from bemo.environment.scenario_generator import ScenarioGenerator, Scenario
from bemo.objectives.base import PathResult


BOUNDS = np.array([[0, 0, 0], [10, 10, 5]], dtype=float)


def make_box(pos=(5, 5, 2), half=(0.5, 0.5, 1.0), obs_id=0):
    return BoxObstacle(obs_id, np.array(pos, float), np.array(half, float))


def make_cylinder(pos=(3, 3, 0), radius=0.5, height=3.0, obs_id=1):
    return CylinderObstacle(obs_id, np.array(pos, float), radius, height)


def make_obstacle_map(obstacles=None):
    if obstacles is None:
        obstacles = [make_box(), make_cylinder()]
    return ObstacleMap(obstacles, BOUNDS)


def make_straight_path(start=(0, 0, 1), end=(8, 8, 1), n=50, algorithm="test"):
    start, end = np.array(start, float), np.array(end, float)
    ts = np.linspace(0, 1, n)
    wps = np.array([start + t * (end - start) for t in ts])
    vels = np.tile((end - start) / n, (n, 1))
    accs = np.zeros((n, 3))
    timestamps = np.linspace(0, 5, n)
    return PathResult(
        waypoints=wps, velocities=vels, accelerations=accs,
        timestamps=timestamps, uav_id=0, reached_target=True, algorithm=algorithm
    )


def make_scenario(n_obstacles=3, n_uavs=1, seed=42):
    gen = ScenarioGenerator(BOUNDS, seed=seed)
    return gen.generate(n_obstacles=n_obstacles, n_uavs=n_uavs)


def make_uav_state(pos=(1, 1, 1)):
    return UAVState(
        position=np.array(pos, float),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        heading=0.0,
        uav_id=0,
    )
