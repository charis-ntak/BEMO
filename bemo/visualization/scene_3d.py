"""3D scene visualization with Matplotlib."""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bemo.environment.world import SimulationWorld
    from bemo.environment.obstacles import BoxObstacle, CylinderObstacle
    from bemo.objectives.base import PathResult

ALGORITHM_COLORS = {
    "RRT*":    "tab:blue",
    "APF-PSO": "tab:orange",
    "PPO":     "tab:green",
    "DDPG":    "tab:red",
}


class Scene3DVisualizer:
    """Matplotlib 3D visualization: obstacles, paths, UAV positions."""

    def __init__(self, bounds: np.ndarray, figsize=(12, 8)):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        self.bounds = np.asarray(bounds, dtype=float)
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._setup_axes()

    def _setup_axes(self):
        lo, hi = self.bounds[0], self.bounds[1]
        self.ax.set_xlim(lo[0], hi[0])
        self.ax.set_ylim(lo[1], hi[1])
        self.ax.set_zlim(lo[2], hi[2])
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_title("UAV Path Planning — 3D Scene")

    def draw_obstacles(self, obstacles, alpha: float = 0.3):
        """Draw all obstacles."""
        from bemo.environment.obstacles import BoxObstacle, CylinderObstacle

        for obs in obstacles:
            if isinstance(obs, BoxObstacle):
                self._draw_box(obs, alpha)
            elif isinstance(obs, CylinderObstacle):
                self._draw_cylinder(obs, alpha)

    def _draw_box(self, obs, alpha: float):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        hx, hy, hz = obs.half_extents
        cx, cy, cz = obs.position

        # 8 corners
        corners = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
        ])
        # Apply rotation and translate
        corners = (obs.rotation @ corners.T).T + obs.position

        # 6 faces
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],  # bottom
            [corners[4], corners[5], corners[6], corners[7]],  # top
            [corners[0], corners[1], corners[5], corners[4]],  # front
            [corners[2], corners[3], corners[7], corners[6]],  # back
            [corners[1], corners[2], corners[6], corners[5]],  # right
            [corners[0], corners[3], corners[7], corners[4]],  # left
        ]
        poly = Poly3DCollection(faces, alpha=alpha, facecolor="gray", edgecolor="darkgray")
        self.ax.add_collection3d(poly)

    def _draw_cylinder(self, obs, alpha: float):
        theta = np.linspace(0, 2 * np.pi, 30)
        z = np.linspace(0, obs.height, 10)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = obs.position[0] + obs.radius * np.cos(theta_grid)
        y_grid = obs.position[1] + obs.radius * np.sin(theta_grid)
        z_grid = obs.position[2] + z_grid
        self.ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha,
                             color="sienna", linewidth=0)

    def draw_path(self, path: "PathResult",
                  color: Optional[str] = None,
                  label: Optional[str] = None,
                  linewidth: float = 2.0,
                  draw_arrows: bool = False):
        """Draw a UAV path."""
        if path.n_steps < 2:
            return
        wp = path.waypoints
        alg = path.algorithm
        c = color or ALGORITHM_COLORS.get(alg, "blue")
        lbl = label or alg

        self.ax.plot(wp[:, 0], wp[:, 1], wp[:, 2],
                     color=c, linewidth=linewidth, label=lbl, alpha=0.8)
        # Mark start and end
        self.ax.scatter(*wp[0], color=c, marker="o", s=50, zorder=5)
        self.ax.scatter(*wp[-1], color=c,
                        marker="*" if path.reached_target else "x",
                        s=100, zorder=5)

    def draw_start_target(self, start: np.ndarray, target: np.ndarray,
                           uav_id: int = 0):
        self.ax.scatter(*start, color="green", marker="^", s=120, zorder=6,
                        label=f"Start UAV {uav_id}")
        self.ax.scatter(*target, color="red", marker="D", s=120, zorder=6,
                        label=f"Target UAV {uav_id}")

    def compare_paths(self, paths: Dict[str, "PathResult"]):
        """Draw multiple algorithm paths with distinct colors."""
        for label, path in paths.items():
            color = ALGORITHM_COLORS.get(label, None)
            self.draw_path(path, color=color, label=label)

    def animate(self, path: "PathResult", interval_ms: int = 50):
        """Animate UAV moving along path."""
        from matplotlib.animation import FuncAnimation

        wp = path.waypoints
        point, = self.ax.plot([], [], [], "ro", markersize=8)

        def update(frame):
            point.set_data([wp[frame, 0]], [wp[frame, 1]])
            point.set_3d_properties([wp[frame, 2]])
            return point,

        anim = FuncAnimation(self.fig, update, frames=len(wp),
                             interval=interval_ms, blit=False)
        return anim

    def show(self):
        import matplotlib.pyplot as plt
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.show()

    def save(self, filepath: str, dpi: int = 150):
        import matplotlib.pyplot as plt
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to {filepath}")
