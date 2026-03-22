"""BenchmarkSuite: run all algorithms on same scenarios and compare."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from bemo.objectives.base import PathResult
from bemo.aggregation.pareto import ParetoFront, Solution, fast_non_dominated_sort
from bemo.aggregation.fuzzy_fis import MamdaniFIS
from .metrics import PathMetrics, compute_metrics


@dataclass
class BenchmarkResults:
    """Collected results from a benchmark run."""
    metrics: List[PathMetrics] = field(default_factory=list)
    paths: List[PathResult] = field(default_factory=list)
    pareto_front: Optional[ParetoFront] = None
    fuzzy_scores: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Mean metrics per algorithm."""
        alg_metrics: Dict[str, List[PathMetrics]] = {}
        for m in self.metrics:
            alg_metrics.setdefault(m.algorithm, []).append(m)

        result = {}
        for alg, ms in alg_metrics.items():
            result[alg] = {
                "path_length_mean": float(np.mean([m.path_length_m for m in ms])),
                "path_length_std": float(np.std([m.path_length_m for m in ms])),
                "energy_mean": float(np.mean([m.energy_proxy for m in ms])),
                "min_clearance_mean": float(np.mean([m.min_clearance_m for m in ms])),
                "curvature_mean": float(np.mean([m.mean_curvature for m in ms])),
                "success_rate": float(np.mean([m.reached_target for m in ms])),
                "planning_time_mean": float(np.mean([m.planning_time_s for m in ms])),
                "fuzzy_fitness_mean": float(np.mean([m.fuzzy_fitness for m in ms if m.fuzzy_fitness is not None] or [0.0])),
            }
        return result


class BenchmarkSuite:
    """Runs all planners on the same set of scenarios and collects metrics."""

    def __init__(self,
                 planners,
                 objectives,
                 pareto_front: Optional[ParetoFront] = None,
                 fis: Optional[MamdaniFIS] = None):
        self.planners = planners
        self.objectives = objectives
        self.pareto_front = pareto_front or ParetoFront()
        self.fis = fis or MamdaniFIS()

    def run(self,
            scenarios,
            n_runs: int = 1,
            output_dir: str = "results/",
            verbose: bool = True) -> BenchmarkResults:
        """Run all planners on all scenarios.

        Args:
            scenarios: list of Scenario objects.
            n_runs: number of independent runs per scenario×planner.
            output_dir: where to save results.
            verbose: print progress.

        Returns:
            BenchmarkResults with metrics, paths, Pareto front, FIS scores.
        """
        from bemo.environment.world import SimulationWorld
        from bemo.environment.uav import UAVDynamics
        from bemo.environment.obstacles import ObstacleMap

        all_metrics: List[PathMetrics] = []
        all_paths: List[PathResult] = []
        all_solutions: List[Solution] = []

        default_dynamics = UAVDynamics()

        for scenario_idx, scenario in enumerate(scenarios):
            scenario_id = f"scenario_{scenario_idx:03d}"
            obstacle_map = ObstacleMap(scenario.obstacles, scenario.bounds)

            if verbose:
                print(f"\nScenario {scenario_idx + 1}/{len(scenarios)} "
                      f"(seed={scenario.seed}, {len(scenario.obstacles)} obstacles, "
                      f"{scenario.n_uavs} UAVs)")

            for uav_idx in range(scenario.n_uavs):
                start = scenario.starts[uav_idx]
                target = scenario.targets[uav_idx]

                for planner in self.planners:
                    for run_i in range(n_runs):
                        if verbose:
                            print(f"  {planner.name()} | UAV {uav_idx} | run {run_i + 1}/{n_runs}",
                                  end=" ... ", flush=True)

                        t_start = time.time()
                        try:
                            path = planner.plan(start, target, obstacle_map,
                                                scenario.bounds, default_dynamics)
                        except Exception as e:
                            if verbose:
                                print(f"FAILED: {e}")
                            continue
                        planning_time = time.time() - t_start

                        if verbose:
                            status = "reached" if path.reached_target else "partial"
                            print(f"{status} ({planning_time:.2f}s)")

                        m = compute_metrics(path, obstacle_map, self.objectives,
                                            planning_time, scenario_id)
                        all_metrics.append(m)
                        all_paths.append(path)

                        # Build solution for Pareto/FIS
                        obj_values = self._compute_objective_vector(path, obstacle_map)
                        sol = Solution(
                            solution_id=f"{scenario_id}_{planner.name()}_uav{uav_idx}_run{run_i}",
                            algorithm=planner.name(),
                            path=path,
                            objectives=obj_values,
                            objective_names=["path_length", "energy", "smoothness", "safety_neg"],
                        )
                        self.pareto_front.add(sol)
                        all_solutions.append(sol)

        # Compute FIS scores across all collected solutions
        if all_solutions:
            fis_scores = self.fis.batch_evaluate(all_solutions, self.objectives, None)
            for sol, score in zip(all_solutions, fis_scores):
                sol_id = sol.solution_id
                # Update metrics with FIS score
                for m in all_metrics:
                    if m.scenario_id in sol_id and m.algorithm == sol.algorithm:
                        m.fuzzy_fitness = score
                        break

        # Assign Pareto ranks
        fronts = fast_non_dominated_sort(all_solutions)
        for rank, front in enumerate(fronts):
            for sol in front:
                for m in all_metrics:
                    if m.algorithm == sol.algorithm and m.scenario_id in sol.solution_id:
                        m.pareto_rank = rank
                        break

        results = BenchmarkResults(
            metrics=all_metrics,
            paths=all_paths,
            pareto_front=self.pareto_front,
        )

        # Save results
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            from bemo.evaluation.reporter import Reporter
            reporter = Reporter(output_dir)
            reporter.save_metrics(all_metrics)
            reporter.save_summary(results.summary())
        except Exception as e:
            warnings.warn(f"Could not save results: {e}")

        return results

    def _compute_objective_vector(self, path: PathResult,
                                   obstacle_map) -> np.ndarray:
        """Compute 4-dim objective vector in all-minimize convention."""
        obj_map = {fn.name: fn for fn in self.objectives}
        length  = obj_map.get("path_length", None)
        energy  = obj_map.get("energy", None)
        smooth  = obj_map.get("smoothness", None)
        safety  = obj_map.get("safety", None)

        v_length = length.evaluate(path, obstacle_map) if length else 0.0
        v_energy = energy.evaluate(path, obstacle_map) if energy else 0.0
        v_smooth = smooth.evaluate(path, obstacle_map) if smooth else 0.0
        v_safety = safety.evaluate(path, obstacle_map) if safety else 0.0

        # Negate safety so all objectives are minimize
        return np.array([v_length, v_energy, v_smooth, -v_safety])
