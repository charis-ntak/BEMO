#!/usr/bin/env python3
"""BEMO UAV Path Planning & Obstacle Avoidance — CLI entry point."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import yaml


def load_config(config_path: str = "config/default_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_planners(algorithm: str, config: dict, planner_cfg=None):
    """Build requested planner(s)."""
    from bemo.algorithms.base import PlannerConfig
    from bemo.algorithms.rrt_star import RRTStarPlanner
    from bemo.algorithms.apf_pso import APFPSOPlanner

    if planner_cfg is None:
        pcfg = config.get("planner", {})
        planner_cfg = PlannerConfig(
            max_iterations=pcfg.get("max_iterations", 5000),
            timeout_seconds=pcfg.get("timeout_seconds", 30.0),
            seed=pcfg.get("seed", 42),
        )

    rrt_cfg = config.get("rrt_star", {})
    apf_cfg = config.get("apf_pso", {})

    planners = {
        "rrt_star": lambda: RRTStarPlanner(
            config=planner_cfg,
            objective_weights=rrt_cfg.get("objective_weights", None),
            step_size=rrt_cfg.get("step_size", 0.3),
            neighbor_radius=rrt_cfg.get("neighbor_radius", 0.8),
            goal_bias=rrt_cfg.get("goal_bias", 0.1),
        ),
        "apf_pso": lambda: APFPSOPlanner(
            config=planner_cfg,
            n_particles=apf_cfg.get("n_particles", 30),
            n_waypoints=apf_cfg.get("n_waypoints", 15),
            sensor_radius=apf_cfg.get("sensor_radius", 2.0),
            k_att=apf_cfg.get("k_att", 1.0),
            k_rep=apf_cfg.get("k_rep", 2.0),
            d0=apf_cfg.get("d0", 1.5),
        ),
    }

    if algorithm == "all":
        return [planners["rrt_star"](), planners["apf_pso"]()]
    elif algorithm in planners:
        return [planners[algorithm]()]
    elif algorithm in ("ppo", "ddpg"):
        print(f"WARNING: {algorithm.upper()} requires a trained model. "
              f"Run: python main.py train --algorithm {algorithm}")
        return []
    else:
        print(f"Unknown algorithm: {algorithm}")
        return []


def build_objectives():
    from bemo.objectives import (PathLengthObjective, EnergyObjective,
                                   SmoothnessObjective, SafetyObjective)
    return [PathLengthObjective(), EnergyObjective(),
            SmoothnessObjective(), SafetyObjective()]


def run_plan(args):
    """Run a single planning task."""
    config = load_config()

    # Override config from args
    config.setdefault("scenario", {})["n_obstacles"] = args.n_obstacles
    config["scenario"]["n_uavs"] = args.n_uavs
    config["scenario"]["seed"] = args.seed

    from bemo.environment.scenario_generator import ScenarioGenerator
    from bemo.environment.obstacles import ObstacleMap
    from bemo.environment.uav import UAVDynamics
    from bemo.objectives import (PathLengthObjective, EnergyObjective,
                                   SmoothnessObjective, SafetyObjective)
    from bemo.aggregation.pareto import ParetoFront, Solution
    from bemo.aggregation.fuzzy_fis import MamdaniFIS
    from bemo.evaluation.metrics import compute_metrics

    bounds = np.array(config["environment"]["bounds"], dtype=float)
    gen = ScenarioGenerator(bounds, seed=args.seed)
    scenario = gen.generate(n_obstacles=args.n_obstacles, n_uavs=args.n_uavs)
    dynamics = UAVDynamics.from_config(config)
    obstacle_map = ObstacleMap(scenario.obstacles, bounds)

    planners = build_planners(args.algorithm, config)
    if not planners:
        return

    objectives = build_objectives()
    pareto_front = ParetoFront()
    fis = MamdaniFIS()

    print(f"\nScenario: {len(scenario.obstacles)} obstacles, {scenario.n_uavs} UAV(s)")
    print(f"Start: {scenario.starts[0].round(2)}, Target: {scenario.targets[0].round(2)}")

    paths = {}
    all_solutions = []
    for planner in planners:
        print(f"\nRunning {planner.name()} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            path = planner.plan(
                scenario.starts[0], scenario.targets[0],
                obstacle_map, bounds, dynamics
            )
            elapsed = time.time() - t0
            status = "REACHED" if path.reached_target else "PARTIAL"
            print(f"{status} in {elapsed:.2f}s | {path.n_steps} steps")

            m = compute_metrics(path, obstacle_map, objectives, elapsed, "single")
            print(f"  Path length: {m.path_length_m:.2f}m | "
                  f"Energy: {m.energy_proxy:.4f} | "
                  f"Min clearance: {m.min_clearance_m:.3f}m | "
                  f"Curvature: {m.mean_curvature:.4f}")

            paths[planner.name()] = path

            # Add to Pareto front
            obj_vec = np.array([
                m.path_length_m, m.energy_proxy, m.mean_curvature, -m.min_clearance_m
            ])
            sol = Solution(
                solution_id=f"{planner.name()}_single",
                algorithm=planner.name(),
                path=path,
                objectives=obj_vec,
            )
            pareto_front.add(sol)
            all_solutions.append(sol)

        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Aggregation results
    if args.aggregation in ("pareto", "both"):
        front = pareto_front.get_front()
        print(f"\nPareto front size: {len(front)}")
        hv = pareto_front.hypervolume()
        print(f"Hypervolume indicator: {hv:.4f}")
        for s in front:
            print(f"  [{s.algorithm}] objectives: {s.objectives.round(4)}")

    if args.aggregation in ("fuzzy", "both") and all_solutions:
        print("\nMamdani FIS fitness scores:")
        fis_scores = fis.batch_evaluate(all_solutions, objectives, obstacle_map)
        for sol, score in zip(all_solutions, fis_scores):
            print(f"  [{sol.algorithm}] fitness = {score:.4f}")

    # Visualization
    if args.visualize and paths:
        try:
            from bemo.visualization.scene_3d import Scene3DVisualizer
            vis = Scene3DVisualizer(bounds)
            vis.draw_obstacles(scenario.obstacles)
            vis.draw_start_target(scenario.starts[0], scenario.targets[0])
            vis.compare_paths(paths)
            if args.save_results:
                vis.save(os.path.join(args.save_results, "scene_3d.png"))
            else:
                vis.show()
        except Exception as e:
            print(f"Visualization failed: {e}")

    if args.save_results:
        from bemo.evaluation.reporter import Reporter
        os.makedirs(args.save_results, exist_ok=True)
        reporter = Reporter(args.save_results)


def run_train(args):
    """Train an RL agent."""
    config = load_config()

    if args.algorithm == "ppo":
        from bemo.algorithms.ppo_planner import PPOPlanner
        planner = PPOPlanner.from_config(config)
        planner.train(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_path=os.path.join(args.save_path, "ppo_uav"),
        )
    elif args.algorithm == "ddpg":
        from bemo.algorithms.ddpg_planner import DDPGPlanner
        planner = DDPGPlanner.from_config(config)
        planner.train(
            total_timesteps=args.timesteps,
            save_path=os.path.join(args.save_path, "ddpg_uav"),
        )
    else:
        print(f"Unknown RL algorithm: {args.algorithm}")


def run_benchmark(args):
    """Run full algorithm comparison benchmark."""
    config = load_config()

    from bemo.environment.scenario_generator import ScenarioGenerator
    from bemo.aggregation.pareto import ParetoFront
    from bemo.aggregation.fuzzy_fis import MamdaniFIS
    from bemo.evaluation.benchmarker import BenchmarkSuite
    from bemo.evaluation.reporter import Reporter
    from bemo.visualization.metrics_dashboard import MetricsDashboard
    from bemo.visualization.pareto_plot import ParetoPlot

    bounds = np.array(config["environment"]["bounds"], dtype=float)
    gen = ScenarioGenerator(bounds, seed=config.get("scenario", {}).get("seed", 42))
    scenarios = [
        gen.generate(
            n_obstacles=config.get("scenario", {}).get("n_obstacles", 5),
            n_uavs=config.get("scenario", {}).get("n_uavs", 1),
        )
        for _ in range(args.n_scenarios)
    ]

    planners = build_planners("all", config)
    objectives = build_objectives()
    pareto_front = ParetoFront()
    fis = MamdaniFIS()

    suite = BenchmarkSuite(planners, objectives, pareto_front, fis)
    results = suite.run(scenarios, n_runs=args.n_runs, output_dir=args.output_dir)

    summary = results.summary()
    reporter = Reporter(args.output_dir)
    reporter.print_summary(summary)
    reporter.save_csv(results.metrics)

    # Visualization
    try:
        dashboard = MetricsDashboard()
        dashboard.bar_comparison(
            summary,
            save_path=os.path.join(args.output_dir, "bar_comparison.png"),
        )
        dashboard.radar_chart(
            summary,
            save_path=os.path.join(args.output_dir, "radar_chart.png"),
        )
        pareto_plot = ParetoPlot()
        pareto_plot.plot_2d(
            results.pareto_front,
            all_solutions=None,
            save_path=os.path.join(args.output_dir, "pareto_2d.png"),
        )
    except Exception as e:
        print(f"Visualization error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="BEMO UAV Path Planning & Obstacle Avoidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py plan --algorithm rrt_star --n-obstacles 5 --visualize
  python main.py plan --algorithm all --aggregation both
  python main.py train --algorithm ppo --timesteps 500000
  python main.py benchmark --n-scenarios 10 --n-runs 3
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # plan subcommand
    plan_p = subparsers.add_parser("plan", help="Run a single planning task")
    plan_p.add_argument("--algorithm", choices=["rrt_star", "apf_pso", "ppo", "ddpg", "all"],
                        default="all")
    plan_p.add_argument("--n-obstacles", type=int, default=5)
    plan_p.add_argument("--n-uavs", type=int, default=1)
    plan_p.add_argument("--seed", type=int, default=42)
    plan_p.add_argument("--aggregation", choices=["pareto", "fuzzy", "both", "none"],
                        default="both")
    plan_p.add_argument("--visualize", action="store_true")
    plan_p.add_argument("--save-results", type=str, default=None)

    # train subcommand
    train_p = subparsers.add_parser("train", help="Train RL agent")
    train_p.add_argument("--algorithm", choices=["ppo", "ddpg"], required=True)
    train_p.add_argument("--timesteps", type=int, default=500_000)
    train_p.add_argument("--n-envs", type=int, default=4)
    train_p.add_argument("--save-path", type=str, default="models/")

    # benchmark subcommand
    bench_p = subparsers.add_parser("benchmark", help="Full algorithm comparison")
    bench_p.add_argument("--n-scenarios", type=int, default=5)
    bench_p.add_argument("--n-runs", type=int, default=3)
    bench_p.add_argument("--output-dir", type=str, default="results/")

    args = parser.parse_args()

    if args.command == "plan":
        run_plan(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
