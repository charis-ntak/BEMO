"""Tests for obstacle SDF geometry."""
import numpy as np
import pytest

from bemo.environment.obstacles import BoxObstacle, CylinderObstacle, ObstacleMap, AABB
from tests.fixtures import BOUNDS, make_box, make_cylinder, make_obstacle_map


class TestBoxObstacle:
    def test_sdf_inside(self):
        obs = make_box(pos=(5, 5, 2), half=(1, 1, 1))
        assert obs.sdf(np.array([5, 5, 2])) < 0  # center is inside

    def test_sdf_outside(self):
        obs = make_box(pos=(5, 5, 2), half=(1, 1, 1))
        assert obs.sdf(np.array([0, 0, 0])) > 0  # far away

    def test_sdf_on_surface(self):
        obs = make_box(pos=(5, 5, 2), half=(1, 1, 1))
        # Point on the face: x=6 (5+1), y=5, z=2
        val = obs.sdf(np.array([6.0, 5.0, 2.0]))
        assert abs(val) < 0.01  # approximately 0 on surface

    def test_sdf_positive_outside(self):
        obs = make_box(pos=(5, 5, 2), half=(0.5, 0.5, 0.5))
        # Point 2 units away along x
        val = obs.sdf(np.array([7.5, 5.0, 2.0]))
        assert abs(val - 2.0) < 0.01

    def test_aabb(self):
        obs = make_box(pos=(5, 5, 2), half=(1, 1, 1))
        aabb = obs.aabb()
        assert np.all(aabb.min_corner <= obs.position)
        assert np.all(aabb.max_corner >= obs.position)

    def test_sample_surface(self):
        obs = make_box(pos=(5, 5, 2), half=(1, 1, 1))
        rng = np.random.default_rng(0)
        pts = obs.sample_surface(100, rng)
        assert pts.shape == (100, 3)
        # All sampled points should have SDF ≈ 0
        sdfs = [abs(obs.sdf(p)) for p in pts[:10]]
        assert all(s < 0.1 for s in sdfs)


class TestCylinderObstacle:
    def test_sdf_inside(self):
        obs = make_cylinder(pos=(3, 3, 0), radius=1.0, height=3.0)
        # Inside: (3, 3, 1.5) — center
        assert obs.sdf(np.array([3, 3, 1.5])) < 0

    def test_sdf_outside_radially(self):
        obs = make_cylinder(pos=(3, 3, 0), radius=1.0, height=3.0)
        # Outside radially: (6, 3, 1.5)
        val = obs.sdf(np.array([6, 3, 1.5]))
        assert val > 0
        assert abs(val - 2.0) < 0.01  # 2m from surface

    def test_sdf_above(self):
        obs = make_cylinder(pos=(3, 3, 0), radius=1.0, height=3.0)
        # Above cylinder: (3, 3, 5) — 2m above top
        val = obs.sdf(np.array([3, 3, 5.0]))
        assert val > 0

    def test_sdf_on_lateral_surface(self):
        obs = make_cylinder(pos=(3, 3, 0), radius=1.0, height=3.0)
        # On lateral surface: (4, 3, 1.5)
        val = obs.sdf(np.array([4.0, 3.0, 1.5]))
        assert abs(val) < 0.01

    def test_aabb(self):
        obs = make_cylinder(pos=(3, 3, 0), radius=1.0, height=3.0)
        aabb = obs.aabb()
        assert aabb.min_corner[0] == pytest.approx(2.0)
        assert aabb.max_corner[2] == pytest.approx(3.0)

    def test_sample_surface(self):
        obs = make_cylinder(pos=(3, 3, 0), radius=1.0, height=3.0)
        pts = obs.sample_surface(50)
        assert pts.shape == (50, 3)


class TestObstacleMap:
    def test_clearance_empty(self):
        omap = ObstacleMap([], BOUNDS)
        assert omap.clearance(np.array([5, 5, 2])) == float("inf")

    def test_clearance_positive_outside(self):
        omap = make_obstacle_map([make_box(pos=(5, 5, 2), half=(0.5, 0.5, 0.5))])
        c = omap.clearance(np.array([0, 0, 0]))
        assert c > 0

    def test_clearance_negative_inside(self):
        omap = make_obstacle_map([make_box(pos=(5, 5, 2), half=(1, 1, 1))])
        c = omap.clearance(np.array([5, 5, 2]))
        assert c < 0

    def test_is_collision_free(self):
        box = make_box(pos=(5, 5, 2), half=(1, 1, 1))
        omap = make_obstacle_map([box])
        assert omap.is_collision_free(np.array([0, 0, 0]))
        assert not omap.is_collision_free(np.array([5, 5, 2]))

    def test_segment_clearance(self):
        box = make_box(pos=(5, 5, 2), half=(1, 1, 1))
        omap = make_obstacle_map([box])
        # Segment through free space
        c = omap.segment_clearance(np.array([0, 0, 0]), np.array([2, 0, 0]))
        assert c > 0

    def test_is_within_bounds(self):
        omap = make_obstacle_map()
        assert omap.is_within_bounds(np.array([5, 5, 2]))
        assert not omap.is_within_bounds(np.array([15, 5, 2]))

    def test_nearest_obstacles(self):
        box = make_box(pos=(5, 5, 2))
        cyl = make_cylinder(pos=(3, 3, 0))
        omap = make_obstacle_map([box, cyl])
        nearest = omap.nearest_obstacles(np.array([0, 0, 0]), k=1)
        assert len(nearest) == 1
        assert nearest[0].id == cyl.id  # cylinder is closer to origin

    def test_nearest_obstacle_gradient(self):
        box = make_box(pos=(5, 5, 2), half=(0.5, 0.5, 0.5))
        omap = make_obstacle_map([box])
        grad = omap.nearest_obstacle_gradient(np.array([0, 0, 0]))
        assert grad.shape == (3,)
        # Gradient should point away from obstacle (toward lower clearance)
        assert np.linalg.norm(grad) > 0
