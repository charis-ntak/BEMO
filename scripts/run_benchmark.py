#!/usr/bin/env python3
"""Full benchmark pipeline: run all algorithms, compare, visualize."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from bemo.environment.scenario_generator import ScenarioGenerator
from bemo.algorithms.rrt_star import RRTStarPlanner
from bemo.algorithms.apf_pso import APFPSOPlanner
from bemo.algorithms.base import PlannerConfig
from bemo.objectives import (PathLengthObjective, EnergyObjective,
                               SmoothnessObjective, SafetyObjective)
from bemo.aggregation.pareto import ParetoFront
from bemo.aggregation.fuzzy_fis import MamdaniFIS
from bemo.evaluation.benchmarker import BenchmarkSuite
from bemo.evaluation.reporter import Reporter
from bemo.visualization.metrics_dashboard import MetricsDashboard
from bemo.visualization.pareto_plot import ParetoPlot


def main():
    parser = argparse.ArgumentParser(description="Run full algorithm benchmark")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--n-scenarios", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--output-dir", default="results/benchmark")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    bounds = np.array(config["environment"]["bounds"], dtype=float)
    gen = ScenarioGenerator(bounds, seed=config.get("scenario", {}).get("seed", 42))
    scenarios = [
        gen.generate(
            n_obstacles=config.get("scenario", {}).get("n_obstacles", 5),
            n_uavs=config.get("scenario", {}).get("n_uavs", 1),
        )
        for _ in range(args.n_scenarios)
    ]

    planner_cfg = PlannerConfig(
        max_iterations=config.get("planner", {}).get("max_iterations", 5000),
        timeout_seconds=config.get("planner", {}).get("timeout_seconds", 30.0),
        seed=config.get("planner", {}).get("seed", 42),
    )
    rrt_cfg = config.get("rrt_star", {})
    apf_cfg = config.get("apf_pso", {})

    planners = [
        RRTStarPlanner(
            config=planner_cfg,
            step_size=rrt_cfg.get("step_size", 0.3),
            neighbor_radius=rrt_cfg.get("neighbor_radius", 0.8),
            goal_bias=rrt_cfg.get("goal_bias", 0.1),
        ),
        APFPSOPlanner(
            config=planner_cfg,
            n_particles=apf_cfg.get("n_particles", 30),
            n_waypoints=apf_cfg.get("n_waypoints", 15),
        ),
    ]

    objectives = [PathLengthObjective(), EnergyObjective(),
                  SmoothnessObjective(), SafetyObjective()]
    pareto_front = ParetoFront()
    fis = MamdaniFIS()

    suite = BenchmarkSuite(planners, objectives, pareto_front, fis)
    results = suite.run(scenarios, n_runs=args.n_runs, output_dir=args.output_dir)
    summary = results.summary()

    reporter = Reporter(args.output_dir)
    reporter.print_summary(summary)
    reporter.save_metrics(results.metrics)
    reporter.save_csv(results.metrics)

    if not args.no_viz:
        dashboard = MetricsDashboard()
        dashboard.bar_comparison(summary,
                                  save_path=os.path.join(args.output_dir, "bar_comparison.png"))
        dashboard.radar_chart(summary,
                              save_path=os.path.join(args.output_dir, "radar_chart.png"))
        pp = ParetoPlot()
        pp.plot_2d(results.pareto_front,
                   save_path=os.path.join(args.output_dir, "pareto_front.png"))
        print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
