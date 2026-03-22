"""Obstacle geometry using Signed Distance Functions (SDF)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree


@dataclass
class AABB:
    """Axis-Aligned Bounding Box for broad-phase collision rejection."""
    min_corner: np.ndarray  # shape (3,)
    max_corner: np.ndarray  # shape (3,)

    def contains(self, point: np.ndarray) -> bool:
        return np.all(point >= self.min_corner) and np.all(point <= self.max_corner)

    def intersects(self, other: "AABB") -> bool:
        return (np.all(self.min_corner <= other.max_corner) and
                np.all(other.min_corner <= self.max_corner))


class Obstacle(ABC):
    """Abstract base for all obstacle types. SDF is the core interface."""

    def __init__(self, obstacle_id: int, position: np.ndarray):
        self.id = obstacle_id
        self.position = np.asarray(position, dtype=float)

    @abstractmethod
    def sdf(self, point: np.ndarray) -> float:
        """Signed Distance Function. Negative = inside obstacle, positive = outside."""
        ...

    @abstractmethod
    def aabb(self) -> AABB:
        """Axis-aligned bounding box for spatial indexing."""
        ...

    @abstractmethod
    def sample_surface(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample n points on the obstacle surface. shape: (n, 3)."""
        ...


class BoxObstacle(Obstacle):
    """Rectangular box obstacle with optional rotation."""

    def __init__(self, obstacle_id: int, position: np.ndarray,
                 half_extents: np.ndarray,
                 rotation: Optional[np.ndarray] = None):
        super().__init__(obstacle_id, position)
        self.half_extents = np.asarray(half_extents, dtype=float)
        self.rotation = rotation if rotation is not None else np.eye(3)

    def sdf(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=float)
        # Transform to obstacle-local frame
        local = self.rotation.T @ (point - self.position)
        q = np.abs(local) - self.half_extents
        return float(np.linalg.norm(np.maximum(q, 0.0)) + min(float(np.max(q)), 0.0))

    def aabb(self) -> AABB:
        # Conservative AABB using max half-extent for rotated box
        max_ext = np.sqrt(np.sum(self.half_extents**2)) * np.ones(3)
        return AABB(self.position - max_ext, self.position + max_ext)

    def sample_surface(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        hx, hy, hz = self.half_extents
        # Areas of 6 faces
        areas = np.array([hy * hz, hy * hz, hx * hz, hx * hz, hx * hy, hx * hy]) * 4
        face_probs = areas / areas.sum()
        faces = rng.choice(6, size=n, p=face_probs)
        points = []
        for f in faces:
            if f == 0:   p = np.array([-hx, rng.uniform(-hy, hy), rng.uniform(-hz, hz)])
            elif f == 1: p = np.array([hx,  rng.uniform(-hy, hy), rng.uniform(-hz, hz)])
            elif f == 2: p = np.array([rng.uniform(-hx, hx), -hy, rng.uniform(-hz, hz)])
            elif f == 3: p = np.array([rng.uniform(-hx, hx),  hy, rng.uniform(-hz, hz)])
            elif f == 4: p = np.array([rng.uniform(-hx, hx), rng.uniform(-hy, hy), -hz])
            else:        p = np.array([rng.uniform(-hx, hx), rng.uniform(-hy, hy),  hz])
            points.append(self.position + self.rotation @ p)
        return np.array(points)

    def __repr__(self) -> str:
        return f"BoxObstacle(id={self.id}, pos={self.position}, extents={self.half_extents})"


class CylinderObstacle(Obstacle):
    """Vertical cylinder obstacle."""

    def __init__(self, obstacle_id: int, position: np.ndarray,
                 radius: float, height: float):
        """
        Args:
            position: base center of the cylinder.
            radius: cylinder radius.
            height: cylinder height (extends upward from position).
        """
        super().__init__(obstacle_id, position)
        self.radius = float(radius)
        self.height = float(height)

    def sdf(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=float)
        rel = point - self.position
        # Radial distance from cylinder axis (negative inside)
        d_radial = float(np.sqrt(rel[0]**2 + rel[1]**2)) - self.radius
        # Signed axial distance from [0, height] slab (positive outside, negative inside)
        d_axial = max(-rel[2], rel[2] - self.height)
        if d_radial > 0 and d_axial > 0:
            return float(np.sqrt(d_radial**2 + d_axial**2))
        return float(max(d_radial, d_axial))

    def aabb(self) -> AABB:
        min_c = self.position + np.array([-self.radius, -self.radius, 0.0])
        max_c = self.position + np.array([self.radius, self.radius, self.height])
        return AABB(min_c, max_c)

    def sample_surface(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        lateral_area = 2 * np.pi * self.radius * self.height
        cap_area = np.pi * self.radius**2
        total = lateral_area + 2 * cap_area
        p_lateral = lateral_area / total
        p_top = p_bottom = cap_area / total

        points = []
        choices = rng.choice(3, size=n, p=[p_lateral, p_top, p_bottom])
        for c in choices:
            if c == 0:  # lateral surface
                theta = rng.uniform(0, 2 * np.pi)
                z = rng.uniform(0, self.height)
                p = self.position + np.array([self.radius * np.cos(theta),
                                              self.radius * np.sin(theta), z])
            elif c == 1:  # top cap
                r = self.radius * np.sqrt(rng.uniform(0, 1))
                theta = rng.uniform(0, 2 * np.pi)
                p = self.position + np.array([r * np.cos(theta), r * np.sin(theta), self.height])
            else:  # bottom cap
                r = self.radius * np.sqrt(rng.uniform(0, 1))
                theta = rng.uniform(0, 2 * np.pi)
                p = self.position + np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
            points.append(p)
        return np.array(points)

    def __repr__(self) -> str:
        return f"CylinderObstacle(id={self.id}, pos={self.position}, r={self.radius}, h={self.height})"


class ObstacleMap:
    """Spatial index over all obstacles for efficient clearance queries."""

    def __init__(self, obstacles: List[Obstacle], bounds: np.ndarray):
        """
        Args:
            obstacles: list of Obstacle instances.
            bounds: shape (2, 3) — [[xmin,ymin,zmin], [xmax,ymax,zmax]].
        """
        self._obstacles = list(obstacles)
        self._bounds = np.asarray(bounds, dtype=float)
        self._rebuild_kdtree()

    def _rebuild_kdtree(self):
        if self._obstacles:
            centroids = np.array([o.position for o in self._obstacles])
            self._kdtree = KDTree(centroids)
        else:
            self._kdtree = None

    @property
    def obstacles(self) -> List[Obstacle]:
        return list(self._obstacles)

    def clearance(self, point: np.ndarray) -> float:
        """Minimum SDF value across all obstacles. Positive = free space."""
        if not self._obstacles:
            return float("inf")
        point = np.asarray(point, dtype=float)
        return float(min(o.sdf(point) for o in self._obstacles))

    def is_collision_free(self, point: np.ndarray, uav_radius: float = 0.15) -> bool:
        return self.clearance(point) > uav_radius

    def segment_clearance(self, p1: np.ndarray, p2: np.ndarray,
                          n_samples: int = 20) -> float:
        """Minimum clearance along a line segment (for RRT* edge validation)."""
        p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
        ts = np.linspace(0.0, 1.0, n_samples)
        min_clear = float("inf")
        for t in ts:
            pt = p1 + t * (p2 - p1)
            c = self.clearance(pt)
            if c < min_clear:
                min_clear = c
        return min_clear

    def nearest_obstacle_gradient(self, point: np.ndarray,
                                   eps: float = 1e-4) -> np.ndarray:
        """Numerical gradient of clearance field (points away from nearest obstacle)."""
        point = np.asarray(point, dtype=float)
        grad = np.zeros(3)
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = eps
            grad[i] = (self.clearance(point + dp) - self.clearance(point - dp)) / (2 * eps)
        return grad

    def nearest_obstacles(self, point: np.ndarray, k: int = 5) -> List[Obstacle]:
        """Return k nearest obstacles by centroid distance."""
        if not self._obstacles or self._kdtree is None:
            return []
        k = min(k, len(self._obstacles))
        point = np.asarray(point, dtype=float)
        _, indices = self._kdtree.query(point, k=k)
        if np.isscalar(indices):
            indices = [indices]
        return [self._obstacles[i] for i in indices]

    def obstacles_within_radius(self, point: np.ndarray,
                                 radius: float) -> List[Obstacle]:
        """Return all obstacles within radius of point (by centroid)."""
        if not self._obstacles or self._kdtree is None:
            return []
        point = np.asarray(point, dtype=float)
        indices = self._kdtree.query_ball_point(point, radius)
        return [self._obstacles[i] for i in indices]

    def is_within_bounds(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=float)
        return (np.all(point >= self._bounds[0]) and
                np.all(point <= self._bounds[1]))
