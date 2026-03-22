"""UAV state and point-mass dynamics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class UAVState:
    """Complete UAV state."""
    position: np.ndarray      # shape (3,) meters
    velocity: np.ndarray      # shape (3,) m/s
    acceleration: np.ndarray  # shape (3,) m/s^2
    heading: float            # radians, yaw in XY plane
    uav_id: int

    def to_vector(self) -> np.ndarray:
        """Flatten to 10-dim vector for RL observations."""
        return np.concatenate([
            self.position,
            self.velocity,
            self.acceleration,
            [self.heading],
            [float(self.uav_id)],
        ]).astype(np.float32)

    @classmethod
    def from_vector(cls, v: np.ndarray, uav_id: int) -> "UAVState":
        return cls(
            position=v[0:3].copy(),
            velocity=v[3:6].copy(),
            acceleration=v[6:9].copy(),
            heading=float(v[9]),
            uav_id=uav_id,
        )

    def copy(self) -> "UAVState":
        return UAVState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            heading=self.heading,
            uav_id=self.uav_id,
        )

    def __repr__(self) -> str:
        return (f"UAVState(id={self.uav_id}, "
                f"pos={self.position.round(3)}, "
                f"vel={self.velocity.round(3)})")


class UAVDynamics:
    """Point-mass dynamics with physical constraints."""

    def __init__(self,
                 max_speed: float = 2.0,
                 max_acceleration: float = 3.0,
                 max_jerk: float = 5.0,
                 dt: float = 0.05):
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        self.dt = dt

    def step(self, state: UAVState, action: np.ndarray) -> UAVState:
        """Apply acceleration action and integrate state with Euler method.

        Args:
            state: current UAV state.
            action: desired acceleration command, shape (3,).

        Returns:
            New UAVState after dt seconds.
        """
        action = np.asarray(action, dtype=float)

        # Clip jerk (change in acceleration)
        jerk = (action - state.acceleration) / self.dt
        jerk_norm = np.linalg.norm(jerk)
        if jerk_norm > self.max_jerk:
            jerk = jerk * self.max_jerk / jerk_norm
        new_acc = state.acceleration + jerk * self.dt

        # Clip acceleration magnitude
        acc_norm = np.linalg.norm(new_acc)
        if acc_norm > self.max_acceleration:
            new_acc = new_acc * self.max_acceleration / acc_norm

        # Integrate velocity
        new_vel = state.velocity + new_acc * self.dt

        # Clip speed
        speed = np.linalg.norm(new_vel)
        if speed > self.max_speed:
            new_vel = new_vel * self.max_speed / speed

        # Integrate position
        new_pos = state.position + new_vel * self.dt

        # Update heading from velocity direction in XY
        new_heading = (float(np.arctan2(new_vel[1], new_vel[0]))
                       if np.linalg.norm(new_vel[:2]) > 1e-6
                       else state.heading)

        return UAVState(
            position=new_pos,
            velocity=new_vel,
            acceleration=new_acc,
            heading=new_heading,
            uav_id=state.uav_id,
        )

    def is_within_bounds(self, state: UAVState, bounds: np.ndarray) -> bool:
        bounds = np.asarray(bounds, dtype=float)
        return (np.all(state.position >= bounds[0]) and
                np.all(state.position <= bounds[1]))

    @classmethod
    def from_config(cls, config: dict) -> "UAVDynamics":
        uav_cfg = config.get("uav", {})
        env_cfg = config.get("environment", {})
        return cls(
            max_speed=uav_cfg.get("max_speed", 2.0),
            max_acceleration=uav_cfg.get("max_acceleration", 3.0),
            max_jerk=uav_cfg.get("max_jerk", 5.0),
            dt=env_cfg.get("dt", 0.05),
        )
