"""Metrics dashboard: bar charts and radar plots for algorithm comparison."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class MetricsDashboard:
    """Comparative visualization of algorithm metrics."""

    def __init__(self, figsize=(14, 10)):
        self.figsize = figsize

    def bar_comparison(self, summary: Dict[str, Dict[str, float]],
                        metrics: Optional[List[str]] = None,
                        save_path: Optional[str] = None):
        """Side-by-side bar chart comparing algorithms on each metric."""
        import matplotlib.pyplot as plt

        if metrics is None:
            metrics = ["path_length_mean", "energy_mean", "min_clearance_mean",
                       "curvature_mean", "success_rate", "fuzzy_fitness_mean"]

        algorithms = list(summary.keys())
        n_metrics = len(metrics)
        n_algs = len(algorithms)

        fig, axes = plt.subplots(1, n_metrics, figsize=self.figsize, sharey=False)
        if n_metrics == 1:
            axes = [axes]

        colors = plt.cm.Set2(np.linspace(0, 1, n_algs))

        for ax, metric in zip(axes, metrics):
            values = [summary[alg].get(metric, 0.0) for alg in algorithms]
            bars = ax.bar(algorithms, values, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_title(metric.replace("_", "\n"), fontsize=9)
            ax.set_xticks(range(n_algs))
            ax.set_xticklabels(algorithms, rotation=30, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.3)

            # Value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        plt.suptitle("Algorithm Comparison", fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Dashboard saved to {save_path}")
        else:
            plt.show()

    def radar_chart(self, summary: Dict[str, Dict[str, float]],
                     metrics: Optional[List[str]] = None,
                     save_path: Optional[str] = None):
        """Radar/spider chart comparing algorithms."""
        import matplotlib.pyplot as plt

        if metrics is None:
            metrics = ["path_length_mean", "energy_mean", "min_clearance_mean",
                       "success_rate", "fuzzy_fitness_mean"]

        algorithms = list(summary.keys())
        n_metrics = len(metrics)

        # Normalize each metric to [0, 1] across algorithms
        metric_vals: Dict[str, List[float]] = {}
        for m in metrics:
            vals = [summary[alg].get(m, 0.0) for alg in algorithms]
            lo, hi = min(vals), max(vals)
            if abs(hi - lo) < 1e-10:
                metric_vals[m] = [0.5] * len(algorithms)
            else:
                metric_vals[m] = [(v - lo) / (hi - lo) for v in vals]

        # Flip metrics where lower is better (normalized 0=best becomes 1 for radar)
        lower_is_better = {"path_length_mean", "energy_mean", "curvature_mean",
                            "planning_time_mean"}
        for m in metrics:
            if m in lower_is_better:
                metric_vals[m] = [1.0 - v for v in metric_vals[m]]

        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # close polygon
        metric_labels = [m.replace("_mean", "").replace("_", "\n") for m in metrics]

        fig, ax = plt.subplots(figsize=(8, 8),
                               subplot_kw=dict(polar=True))
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))

        for alg, color in zip(algorithms, colors):
            alg_idx = algorithms.index(alg)
            values = [metric_vals[m][alg_idx] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, color=color, linewidth=2, label=alg)
            ax.fill(angles, values, color=color, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("Algorithm Radar Comparison\n(higher = better)", fontsize=11, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Radar chart saved to {save_path}")
        else:
            plt.show()
