"""Pareto front visualization."""
from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bemo.aggregation.pareto import ParetoFront, Solution


class ParetoPlot:
    """2D and 3D scatter plots of Pareto fronts."""

    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize

    def plot_2d(self, pareto_front: "ParetoFront",
                x_idx: int = 0, y_idx: int = 3,
                all_solutions: Optional[List["Solution"]] = None,
                save_path: Optional[str] = None):
        """2D Pareto front scatter.

        Args:
            x_idx: objective index for x-axis.
            y_idx: objective index for y-axis.
            all_solutions: if provided, plot dominated solutions in gray.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize)
        front = pareto_front.get_front()
        obj_names = pareto_front.objective_names

        if all_solutions:
            front_ids = {s.solution_id for s in front}
            dominated = [s for s in all_solutions if s.solution_id not in front_ids]
            if dominated:
                dom_x = [s.objectives[x_idx] for s in dominated]
                dom_y = [s.objectives[y_idx] for s in dominated]
                ax.scatter(dom_x, dom_y, c="lightgray", s=30, alpha=0.5,
                           label="Dominated", zorder=1)

        if front:
            # Color by algorithm
            algorithms = list({s.algorithm for s in front})
            colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
            alg_color = dict(zip(algorithms, colors))

            for alg in algorithms:
                alg_sols = [s for s in front if s.algorithm == alg]
                x_vals = [s.objectives[x_idx] for s in alg_sols]
                y_vals = [s.objectives[y_idx] for s in alg_sols]
                ax.scatter(x_vals, y_vals, c=[alg_color[alg]], s=80,
                           label=f"{alg} (Pareto)", zorder=3, edgecolors="black", linewidths=0.5)

        x_name = obj_names[x_idx] if x_idx < len(obj_names) else f"obj_{x_idx}"
        y_name = obj_names[y_idx] if y_idx < len(obj_names) else f"obj_{y_idx}"
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"Pareto Front: {x_name} vs {y_name}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Pareto plot saved to {save_path}")
        else:
            plt.show()

    def plot_3d(self, pareto_front: "ParetoFront",
                indices: Tuple[int, int, int] = (0, 1, 3),
                save_path: Optional[str] = None):
        """3D Pareto front scatter."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        front = pareto_front.get_front()
        obj_names = pareto_front.objective_names
        xi, yi, zi = indices

        if front:
            algorithms = list({s.algorithm for s in front})
            colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
            alg_color = dict(zip(algorithms, colors))

            for alg in algorithms:
                alg_sols = [s for s in front if s.algorithm == alg]
                x_vals = [s.objectives[xi] for s in alg_sols]
                y_vals = [s.objectives[yi] for s in alg_sols]
                z_vals = [s.objectives[zi] for s in alg_sols]
                ax.scatter(x_vals, y_vals, z_vals, c=[alg_color[alg]],
                           s=80, label=alg, edgecolors="black", linewidths=0.5)

        ax.set_xlabel(obj_names[xi] if xi < len(obj_names) else f"obj_{xi}")
        ax.set_ylabel(obj_names[yi] if yi < len(obj_names) else f"obj_{yi}")
        ax.set_zlabel(obj_names[zi] if zi < len(obj_names) else f"obj_{zi}")
        ax.set_title("3D Pareto Front")
        ax.legend(loc="best")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
