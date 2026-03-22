#!/usr/bin/env python3
"""
run_reports.py — Run BEMO benchmark and tests, then export results to Excel + PNG.

Usage:
    python run_reports.py [--n-scenarios N] [--n-runs N] [--output-dir DIR] [--skip-tests]

Outputs (in --output-dir, default: reports/):
    metrics.xlsx         Multi-sheet Excel workbook with all results
    scene_3d.png         3D scene: obstacles + algorithm paths
    pareto_front.png     Pareto front scatter (2D projections)
    bar_comparison.png   Side-by-side bar chart of key metrics
    radar_chart.png      Radar/spider chart comparing algorithms
    test_results.txt     pytest output
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PNG saving
import matplotlib.pyplot as plt
import numpy as np

# Ensure BEMO package is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run BEMO benchmark and export reports")
    p.add_argument("--n-scenarios", type=int, default=5,
                   help="Number of random scenarios to benchmark (default: 5)")
    p.add_argument("--n-runs", type=int, default=3,
                   help="Runs per scenario × algorithm (default: 3)")
    p.add_argument("--output-dir", default="reports",
                   help="Output directory for all files (default: reports/)")
    p.add_argument("--skip-tests", action="store_true",
                   help="Skip pytest run")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for scenario generation (default: 42)")
    p.add_argument("--fast", action="store_true",
                   help="Use reduced planner settings for quick demo runs")
    return p.parse_args()


# ─── TEST RUNNER ──────────────────────────────────────────────────────────────

def run_tests(output_dir: str) -> bool:
    """Run pytest and save output to text file. Returns True if all pass."""
    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    report_path = os.path.join(output_dir, "test_results.txt")
    with open(report_path, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
    print(result.stdout[-2000:])  # show last 2000 chars
    print(f"Test results saved to: {report_path}")
    return result.returncode == 0


# ─── BENCHMARK ────────────────────────────────────────────────────────────────

def run_benchmark(n_scenarios: int, n_runs: int, seed: int, fast: bool = False):
    """Run RRT* and APF-PSO benchmark across multiple scenarios."""
    import yaml
    from bemo.environment.scenario_generator import ScenarioGenerator
    from bemo.environment.obstacles import ObstacleMap
    from bemo.environment.uav import UAVDynamics
    from bemo.algorithms.base import PlannerConfig
    from bemo.algorithms.rrt_star import RRTStarPlanner
    from bemo.algorithms.apf_pso import APFPSOPlanner
    from bemo.objectives import (PathLengthObjective, EnergyObjective,
                                   SmoothnessObjective, SafetyObjective)
    from bemo.aggregation.pareto import ParetoFront
    from bemo.aggregation.fuzzy_fis import MamdaniFIS
    from bemo.evaluation.benchmarker import BenchmarkSuite

    with open("config/default_config.yaml") as f:
        config = yaml.safe_load(f)

    bounds = np.array(config["environment"]["bounds"], dtype=float)
    gen = ScenarioGenerator(bounds, seed=seed)
    scenarios = [
        gen.generate(
            n_obstacles=config["scenario"]["n_obstacles"],
            n_uavs=config["scenario"]["n_uavs"],
        )
        for _ in range(n_scenarios)
    ]

    if fast:
        planner_cfg = PlannerConfig(max_iterations=800, timeout_seconds=8.0, seed=seed)
        rrt_step, rrt_radius, rrt_bias = 0.6, 1.2, 0.2
        apf_particles, apf_wps = 15, 10
    else:
        planner_cfg = PlannerConfig(
            max_iterations=config["planner"]["max_iterations"],
            timeout_seconds=config["planner"]["timeout_seconds"],
            seed=seed,
        )
        rrt_cfg = config["rrt_star"]
        apf_cfg = config["apf_pso"]
        rrt_step  = rrt_cfg["step_size"]
        rrt_radius = rrt_cfg["neighbor_radius"]
        rrt_bias  = rrt_cfg["goal_bias"]
        apf_particles = apf_cfg["n_particles"]
        apf_wps    = apf_cfg["n_waypoints"]

    planners = [
        RRTStarPlanner(
            config=planner_cfg,
            step_size=rrt_step,
            neighbor_radius=rrt_radius,
            goal_bias=rrt_bias,
        ),
        APFPSOPlanner(
            config=planner_cfg,
            n_particles=apf_particles,
            n_waypoints=apf_wps,
            sensor_radius=config["apf_pso"]["sensor_radius"],
        ),
    ]

    objectives = [PathLengthObjective(), EnergyObjective(),
                  SmoothnessObjective(), SafetyObjective()]
    pareto_front = ParetoFront()
    fis = MamdaniFIS()

    suite = BenchmarkSuite(planners, objectives, pareto_front, fis)

    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK")
    print("=" * 60)
    results = suite.run(scenarios, n_runs=n_runs,
                        output_dir="results/_temp", verbose=True)
    return results, scenarios, objectives


# ─── EXCEL EXPORT ─────────────────────────────────────────────────────────────

def _apply_header_style(ws, row: int, col_count: int):
    """Apply bold + colored header style to a worksheet row."""
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    header_fill = PatternFill("solid", fgColor="1F497D")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    border = Border(
        bottom=Side(style="medium", color="000000"),
    )
    for col in range(1, col_count + 1):
        cell = ws.cell(row=row, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = border


def _alternate_row_fill(ws, data_start_row: int, data_end_row: int, col_count: int):
    from openpyxl.styles import PatternFill
    alt_fill = PatternFill("solid", fgColor="DCE6F1")
    for row in range(data_start_row, data_end_row + 1):
        if row % 2 == 0:
            for col in range(1, col_count + 1):
                ws.cell(row=row, column=col).fill = alt_fill


def _auto_width(ws):
    """Auto-size column widths based on content."""
    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            try:
                max_len = max(max_len, len(str(cell.value or "")))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 35)


def export_excel(results, scenarios, objectives, output_dir: str):
    """Export all benchmark results to a multi-sheet Excel workbook."""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, numbers
    from openpyxl.chart import BarChart, RadarChart, Reference
    from openpyxl.chart.series import DataPoint
    from openpyxl.utils import get_column_letter

    path = os.path.join(output_dir, "metrics.xlsx")
    wb = openpyxl.Workbook()

    summary = results.summary()

    # ── Sheet 1: Summary ──────────────────────────────────────────────────────
    ws_sum = wb.active
    ws_sum.title = "Summary"

    title_font = Font(bold=True, size=14, color="1F497D")
    ws_sum["A1"] = "BEMO — Algorithm Comparison Summary"
    ws_sum["A1"].font = title_font
    ws_sum["A2"] = f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    ws_sum["A2"].font = Font(italic=True, color="666666")
    ws_sum["A3"] = f"Scenarios: {len(scenarios)}  |  Runs/scenario: {results.metrics[0].scenario_id and len([m for m in results.metrics if m.algorithm == list(summary.keys())[0]])}"
    ws_sum["A3"].font = Font(italic=True, color="666666")

    metric_labels = {
        "path_length_mean":   ("Path Length (m)", "0.00"),
        "path_length_std":    ("Path Length Std",  "0.00"),
        "energy_mean":        ("Energy Proxy",      "0.0000"),
        "min_clearance_mean": ("Min Clearance (m)", "0.000"),
        "curvature_mean":     ("Mean Curvature",    "0.0000"),
        "success_rate":       ("Success Rate",       "0%"),
        "planning_time_mean": ("Planning Time (s)", "0.00"),
        "fuzzy_fitness_mean": ("FIS Fitness",        "0.000"),
    }

    algorithms = list(summary.keys())
    headers = ["Metric"] + algorithms
    ROW = 5
    for col, h in enumerate(headers, 1):
        ws_sum.cell(row=ROW, column=col, value=h)
    _apply_header_style(ws_sum, ROW, len(headers))

    for r, (key, (label, fmt)) in enumerate(metric_labels.items(), 1):
        ws_sum.cell(row=ROW + r, column=1, value=label)
        ws_sum.cell(row=ROW + r, column=1).font = Font(bold=True)
        for c, alg in enumerate(algorithms, 2):
            val = summary[alg].get(key, 0.0)
            cell = ws_sum.cell(row=ROW + r, column=c, value=val)
            if "%" in fmt:
                cell.number_format = "0.0%"
            else:
                cell.number_format = fmt

    _alternate_row_fill(ws_sum, ROW + 1, ROW + len(metric_labels), len(headers))
    _auto_width(ws_sum)

    # Add a bar chart for path length
    bar = BarChart()
    bar.type = "col"
    bar.title = "Path Length (mean)"
    bar.y_axis.title = "meters"
    bar.x_axis.title = "Algorithm"
    bar.width = 15
    bar.height = 10
    # Row 6 = path_length_mean (ROW + 1)
    data_ref = Reference(ws_sum, min_col=2, max_col=1 + len(algorithms),
                          min_row=ROW + 1, max_row=ROW + 1)
    cats = Reference(ws_sum, min_col=2, max_col=1 + len(algorithms),
                     min_row=ROW)
    bar.add_data(data_ref)
    bar.set_categories(cats)
    ws_sum.add_chart(bar, "A15")

    # ── Sheet 2: Per-Run Metrics ───────────────────────────────────────────────
    ws_metrics = wb.create_sheet("Per-Run Metrics")
    columns = ["Algorithm", "Scenario", "Path Length (m)", "Energy",
               "Mean Curvature", "Min Clearance (m)", "Reached Target",
               "Collisions", "Planning Time (s)", "Path Duration (s)",
               "N Waypoints", "Pareto Rank", "FIS Fitness"]
    for col, h in enumerate(columns, 1):
        ws_metrics.cell(row=1, column=col, value=h)
    _apply_header_style(ws_metrics, 1, len(columns))

    for r, m in enumerate(results.metrics, 2):
        row_data = [
            m.algorithm, m.scenario_id,
            round(m.path_length_m, 4), round(m.energy_proxy, 6),
            round(m.mean_curvature, 6), round(m.min_clearance_m, 4),
            m.reached_target, m.n_collisions,
            round(m.planning_time_s, 4), round(m.path_duration_s, 4),
            m.n_waypoints, m.pareto_rank,
            round(m.fuzzy_fitness, 4) if m.fuzzy_fitness is not None else None,
        ]
        for col, val in enumerate(row_data, 1):
            ws_metrics.cell(row=r, column=col, value=val)

    _alternate_row_fill(ws_metrics, 2, 1 + len(results.metrics), len(columns))
    _auto_width(ws_metrics)

    # ── Sheet 3: Pareto Front ─────────────────────────────────────────────────
    ws_pareto = wb.create_sheet("Pareto Front")
    ws_pareto["A1"] = "Pareto-Optimal Solutions"
    ws_pareto["A1"].font = Font(bold=True, size=12, color="1F497D")
    ws_pareto["A2"] = f"Front size: {results.pareto_front.size()}"

    pf_headers = ["Solution ID", "Algorithm", "Path Length", "Energy",
                  "Smoothness", "Safety (neg)", "Hypervolume"]
    for col, h in enumerate(pf_headers, 1):
        ws_pareto.cell(row=4, column=col, value=h)
    _apply_header_style(ws_pareto, 4, len(pf_headers))

    hv = results.pareto_front.hypervolume()
    for r, sol in enumerate(results.pareto_front.get_front(), 5):
        row_data = [
            sol.solution_id, sol.algorithm,
            round(sol.objectives[0], 4) if len(sol.objectives) > 0 else "",
            round(sol.objectives[1], 4) if len(sol.objectives) > 1 else "",
            round(sol.objectives[2], 4) if len(sol.objectives) > 2 else "",
            round(sol.objectives[3], 4) if len(sol.objectives) > 3 else "",
            round(hv, 4) if r == 5 else "",  # only on first row
        ]
        for col, val in enumerate(row_data, 1):
            ws_pareto.cell(row=r, column=col, value=val)

    _alternate_row_fill(ws_pareto, 5, 4 + max(results.pareto_front.size(), 1),
                        len(pf_headers))
    _auto_width(ws_pareto)

    # ── Sheet 4: Per-Scenario Summary ─────────────────────────────────────────
    ws_scen = wb.create_sheet("Per-Scenario")
    ws_scen["A1"] = "Per-Scenario Results"
    ws_scen["A1"].font = Font(bold=True, size=12, color="1F497D")

    scen_headers = ["Scenario", "N Obstacles", "N UAVs"] + \
                   [f"{alg} Length" for alg in algorithms] + \
                   [f"{alg} Clearance" for alg in algorithms] + \
                   [f"{alg} Success" for alg in algorithms]
    for col, h in enumerate(scen_headers, 1):
        ws_scen.cell(row=3, column=col, value=h)
    _apply_header_style(ws_scen, 3, len(scen_headers))

    # Group metrics by scenario
    scen_groups: Dict[str, Dict[str, List]] = {}
    for m in results.metrics:
        scen_groups.setdefault(m.scenario_id, {}).setdefault(m.algorithm, []).append(m)

    for r, (scen_id, alg_data) in enumerate(sorted(scen_groups.items()), 4):
        scen_idx = int(scen_id.split("_")[-1]) if "_" in scen_id else r - 4
        scenario = scenarios[scen_idx] if scen_idx < len(scenarios) else scenarios[0]
        row = [scen_id, len(scenario.obstacles), scenario.n_uavs]
        for alg in algorithms:
            ms = alg_data.get(alg, [])
            row.append(round(np.mean([m.path_length_m for m in ms]), 2) if ms else "N/A")
        for alg in algorithms:
            ms = alg_data.get(alg, [])
            row.append(round(np.mean([m.min_clearance_m for m in ms]), 3) if ms else "N/A")
        for alg in algorithms:
            ms = alg_data.get(alg, [])
            row.append(f"{np.mean([m.reached_target for m in ms]):.0%}" if ms else "N/A")
        for col, val in enumerate(row, 1):
            ws_scen.cell(row=r, column=col, value=val)

    _alternate_row_fill(ws_scen, 4, 3 + len(scen_groups), len(scen_headers))
    _auto_width(ws_scen)

    # ── Sheet 5: Objectives Reference ─────────────────────────────────────────
    ws_obj = wb.create_sheet("Objectives Reference")
    ws_obj["A1"] = "Objective Function Reference"
    ws_obj["A1"].font = Font(bold=True, size=12, color="1F497D")

    obj_headers = ["Objective", "Formula", "Direction", "Notes"]
    for col, h in enumerate(obj_headers, 1):
        ws_obj.cell(row=3, column=col, value=h)
    _apply_header_style(ws_obj, 3, len(obj_headers))

    obj_data = [
        ("Path Length", "Σ ‖p_{t+1} − p_t‖", "Minimize", "Total Euclidean arc length (m)"),
        ("Energy", "∫ ‖a‖² dt", "Minimize", "Integrated squared acceleration (thrust proxy)"),
        ("Smoothness", "mean(‖v × a‖ / ‖v‖³)", "Minimize", "Mean curvature κ along trajectory"),
        ("Safety", "min_t SDF(p_t)", "Maximize", "Min obstacle clearance over full path (m)"),
    ]
    for r, row in enumerate(obj_data, 4):
        for col, val in enumerate(row, 1):
            ws_obj.cell(row=r, column=col, value=val)
    _alternate_row_fill(ws_obj, 4, 3 + len(obj_data), len(obj_headers))
    _auto_width(ws_obj)

    wb.save(path)
    print(f"\nExcel workbook saved: {path}")
    print(f"  Sheets: {', '.join(ws.title for ws in wb.worksheets)}")
    return path


# ─── PNG PLOTS ────────────────────────────────────────────────────────────────

def save_plots(results, scenarios, objectives, output_dir: str):
    """Generate and save all PNG plots."""
    summary = results.summary()
    algorithms = list(summary.keys())
    saved = []

    # ── 1. Bar comparison ─────────────────────────────────────────────────────
    metrics_to_plot = [
        ("path_length_mean", "Path Length (m)", False),
        ("energy_mean",       "Energy",          False),
        ("min_clearance_mean","Min Clearance (m)",True),
        ("curvature_mean",    "Mean Curvature",  False),
        ("success_rate",      "Success Rate",    True),
        ("fuzzy_fitness_mean","FIS Fitness",     True),
    ]
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(16, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
    for ax, (key, label, higher_better) in zip(axes, metrics_to_plot):
        vals = [summary[a].get(key, 0.0) for a in algorithms]
        bars = ax.bar(algorithms, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="y", labelsize=7)
        arrow = "↑" if higher_better else "↓"
        ax.set_xlabel(arrow, fontsize=10, color="green" if higher_better else "red")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Algorithm Comparison — Key Metrics", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(output_dir, "bar_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)
    print(f"  Saved: {p}")

    # ── 2. Radar chart ────────────────────────────────────────────────────────
    radar_metrics = ["path_length_mean", "energy_mean", "min_clearance_mean",
                     "curvature_mean", "success_rate", "fuzzy_fitness_mean"]
    radar_labels = ["Path Length", "Energy", "Clearance", "Curvature", "Success", "FIS Fitness"]
    lower_is_better = {"path_length_mean", "energy_mean", "curvature_mean"}

    norm_vals: Dict[str, List[float]] = {}
    for m in radar_metrics:
        vals = [summary[a].get(m, 0.0) for a in algorithms]
        lo, hi = min(vals), max(vals)
        if abs(hi - lo) < 1e-10:
            norm_vals[m] = [0.5] * len(algorithms)
        else:
            norm_vals[m] = [(v - lo) / (hi - lo) for v in vals]
        if m in lower_is_better:
            norm_vals[m] = [1.0 - x for x in norm_vals[m]]

    n_ax = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7)
    ax.grid(True, alpha=0.3)

    colors_r = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    for alg, color in zip(algorithms, colors_r):
        idx = algorithms.index(alg)
        vals = [norm_vals[m][idx] for m in radar_metrics] + [norm_vals[radar_metrics[0]][idx]]
        ax.plot(angles, vals, color=color, linewidth=2, label=alg)
        ax.fill(angles, vals, color=color, alpha=0.1)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("Algorithm Radar Comparison\n(higher = better)", size=12,
                 fontweight="bold", pad=20)
    plt.tight_layout()
    p = os.path.join(output_dir, "radar_chart.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)
    print(f"  Saved: {p}")

    # ── 3. Pareto front scatter ────────────────────────────────────────────────
    front = results.pareto_front.get_front()
    if front:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        obj_names = ["Path Length", "Energy", "Smoothness", "Safety (neg)"]
        pairs = [(0, 3), (0, 1), (1, 3)]
        all_solutions_by_alg: Dict[str, List] = {}
        for sol in front:
            all_solutions_by_alg.setdefault(sol.algorithm, []).append(sol)

        colors_p = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        alg_color = dict(zip(algorithms, colors_p))

        for ax, (xi, yi) in zip(axes, pairs):
            for alg, sols in all_solutions_by_alg.items():
                xs = [s.objectives[xi] for s in sols]
                ys = [s.objectives[yi] for s in sols]
                ax.scatter(xs, ys, color=alg_color.get(alg, "blue"),
                           s=80, label=alg, edgecolors="black", linewidths=0.5, zorder=3)
            ax.set_xlabel(obj_names[xi], fontsize=9)
            ax.set_ylabel(obj_names[yi], fontsize=9)
            ax.set_title(f"Pareto: {obj_names[xi]} vs {obj_names[yi]}", fontsize=9, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")

        hv = results.pareto_front.hypervolume()
        fig.suptitle(f"Pareto-Optimal Solutions  (front size={results.pareto_front.size()}, "
                     f"hypervolume={hv:.4f})",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        p = os.path.join(output_dir, "pareto_front.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(p)
        print(f"  Saved: {p}")

    # ── 4. 3D scene with example paths ────────────────────────────────────────
    try:
        _save_3d_scene(results, scenarios, output_dir, saved)
    except Exception as e:
        print(f"  WARNING: Could not generate 3D scene: {e}")

    # ── 5. Per-objective distributions (box plot) ─────────────────────────────
    _save_boxplots(results, output_dir, saved)

    return saved


def _save_3d_scene(results, scenarios, output_dir: str, saved: list):
    """3D scene for first scenario with all algorithm paths."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    scenario = scenarios[0]
    bounds = scenario.bounds
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Draw obstacles
    from bemo.environment.obstacles import BoxObstacle, CylinderObstacle
    for obs in scenario.obstacles:
        if isinstance(obs, BoxObstacle):
            hx, hy, hz = obs.half_extents
            cx, cy, cz = obs.position
            corners = np.array([
                [-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],
                [-hx,-hy,hz],[hx,-hy,hz],[hx,hy,hz],[-hx,hy,hz]
            ]) + obs.position
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            faces = [
                [corners[0],corners[1],corners[2],corners[3]],
                [corners[4],corners[5],corners[6],corners[7]],
                [corners[0],corners[1],corners[5],corners[4]],
                [corners[2],corners[3],corners[7],corners[6]],
                [corners[1],corners[2],corners[6],corners[5]],
                [corners[0],corners[3],corners[7],corners[4]],
            ]
            ax.add_collection3d(Poly3DCollection(faces, alpha=0.25,
                                                  facecolor="gray", edgecolor="darkgray"))
        elif isinstance(obs, CylinderObstacle):
            theta = np.linspace(0, 2*np.pi, 20)
            z = np.linspace(0, obs.height, 8)
            T, Z = np.meshgrid(theta, z)
            X = obs.position[0] + obs.radius * np.cos(T)
            Y = obs.position[1] + obs.radius * np.sin(T)
            Z2 = obs.position[2] + Z
            ax.plot_surface(X, Y, Z2, alpha=0.25, color="sienna", linewidth=0)

    # Draw paths from first-scenario metrics (use stored paths)
    colors_map = {"RRT*": "tab:blue", "APF-PSO": "tab:orange",
                  "PPO": "tab:green", "DDPG": "tab:red"}
    plotted_algs = set()
    for path in results.paths:
        if path.n_steps < 2:
            continue
        alg = path.algorithm
        c = colors_map.get(alg, "purple")
        lbl = alg if alg not in plotted_algs else None
        wp = path.waypoints
        ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], color=c, linewidth=2,
                label=lbl, alpha=0.75)
        if lbl:
            ax.scatter(*wp[0], color=c, marker="o", s=40, zorder=5)
            ax.scatter(*wp[-1], color=c,
                       marker="*" if path.reached_target else "x",
                       s=80, zorder=5)
        plotted_algs.add(alg)

    # Mark start/target
    if scenario.starts:
        ax.scatter(*scenario.starts[0], color="lime", marker="^", s=150,
                   zorder=6, label="Start")
    if scenario.targets:
        ax.scatter(*scenario.targets[0], color="red", marker="D", s=150,
                   zorder=6, label="Target")

    ax.set_xlim(bounds[0, 0], bounds[1, 0])
    ax.set_ylim(bounds[0, 1], bounds[1, 1])
    ax.set_zlim(bounds[0, 2], bounds[1, 2])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Scene — Obstacles & Algorithm Paths", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    p = os.path.join(output_dir, "scene_3d.png")
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)
    print(f"  Saved: {p}")


def _save_boxplots(results, output_dir: str, saved: list):
    """Box plots showing distribution of each metric per algorithm."""
    from collections import defaultdict

    alg_metrics: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for m in results.metrics:
        alg_metrics[m.algorithm]["path_length"].append(m.path_length_m)
        alg_metrics[m.algorithm]["energy"].append(m.energy_proxy)
        alg_metrics[m.algorithm]["clearance"].append(m.min_clearance_m)
        alg_metrics[m.algorithm]["curvature"].append(m.mean_curvature)

    algorithms = sorted(alg_metrics.keys())
    metric_keys = [("path_length", "Path Length (m)"),
                   ("energy",       "Energy"),
                   ("clearance",    "Min Clearance (m)"),
                   ("curvature",    "Mean Curvature")]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))

    for ax, (key, label) in zip(axes, metric_keys):
        data = [alg_metrics[a][key] for a in algorithms]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(algorithms, rotation=20, ha="right", fontsize=8)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Metric Distributions per Algorithm", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(output_dir, "boxplots.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)
    print(f"  Saved: {p}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nOutput directory: {os.path.abspath(args.output_dir)}")
    t_total = time.time()

    # 1. Tests
    tests_ok = True
    if not args.skip_tests:
        tests_ok = run_tests(args.output_dir)

    # 2. Benchmark
    results, scenarios, objectives = run_benchmark(
        args.n_scenarios, args.n_runs, args.seed, fast=args.fast
    )

    # 3. Excel report
    print("\n" + "=" * 60)
    print("EXPORTING EXCEL REPORT")
    print("=" * 60)
    try:
        excel_path = export_excel(results, scenarios, objectives, args.output_dir)
    except ImportError:
        print("WARNING: openpyxl not installed. Run: pip install openpyxl")
        excel_path = None

    # 4. PNG plots
    print("\n" + "=" * 60)
    print("SAVING PLOTS")
    print("=" * 60)
    saved_plots = save_plots(results, scenarios, objectives, args.output_dir)

    # 5. Summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    summary = results.summary()
    for alg, stats in summary.items():
        print(f"\n{alg}:")
        print(f"  Path length: {stats['path_length_mean']:.2f} ± {stats['path_length_std']:.2f} m")
        print(f"  Min clearance: {stats['min_clearance_mean']:.3f} m")
        print(f"  Success rate: {stats['success_rate']:.0%}")
        print(f"  FIS fitness: {stats['fuzzy_fitness_mean']:.3f}")

    print(f"\nPareto front: {results.pareto_front.size()} solutions, "
          f"HV={results.pareto_front.hypervolume():.4f}")
    print(f"\nAll outputs saved to: {os.path.abspath(args.output_dir)}/")
    print(f"Total time: {time.time() - t_total:.1f}s")

    if not tests_ok:
        print("\nWARNING: Some tests failed. See test_results.txt for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
