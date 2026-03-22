"""Mamdani Fuzzy Inference System for multi-objective fitness aggregation."""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bemo.aggregation.pareto import Solution
    from bemo.objectives.base import ObjectiveFunction

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False


class MamdaniFIS:
    """Mamdani Fuzzy Inference System aggregating 4 normalized objectives.

    Inputs (antecedents), all normalized to [0, 1] where 1.0 = best:
        - path_quality:   1 - normalized(path_length)
        - energy_quality: 1 - normalized(energy)
        - smoothness:     1 - normalized(curvature)
        - safety:         normalized(min_clearance)

    Output (consequent):
        - fitness: [0, 1] — overall path quality score

    Falls back to a weighted average if scikit-fuzzy is not installed.
    """

    def __init__(self):
        self._use_skfuzzy = SKFUZZY_AVAILABLE
        if self._use_skfuzzy:
            self._build_system()
        else:
            print("WARNING: scikit-fuzzy not available. Using weighted average fallback.")

    def _build_system(self):
        """Build scikit-fuzzy control system."""
        universe = np.linspace(0, 1, 101)

        # Antecedents
        self._path_quality   = ctrl.Antecedent(universe, "path_quality")
        self._energy_quality = ctrl.Antecedent(universe, "energy_quality")
        self._smoothness     = ctrl.Antecedent(universe, "smoothness")
        self._safety         = ctrl.Antecedent(universe, "safety")
        self._fitness        = ctrl.Consequent(universe, "fitness",
                                               defuzzify_method="centroid")

        # Membership functions — trapezoidal for boundaries, triangular for middle
        for antecedent in [self._path_quality, self._energy_quality,
                           self._smoothness, self._safety]:
            antecedent["poor"]       = fuzz.trapmf(antecedent.universe, [0, 0, 0.25, 0.45])
            antecedent["acceptable"] = fuzz.trimf(antecedent.universe, [0.3, 0.5, 0.7])
            antecedent["good"]       = fuzz.trapmf(antecedent.universe, [0.55, 0.75, 1, 1])

        self._fitness["very_poor"]  = fuzz.trapmf(self._fitness.universe, [0, 0, 0.1, 0.25])
        self._fitness["poor"]       = fuzz.trimf(self._fitness.universe, [0.1, 0.25, 0.45])
        self._fitness["acceptable"] = fuzz.trimf(self._fitness.universe, [0.3, 0.5, 0.7])
        self._fitness["good"]       = fuzz.trimf(self._fitness.universe, [0.55, 0.75, 0.9])
        self._fitness["excellent"]  = fuzz.trapmf(self._fitness.universe, [0.75, 0.9, 1, 1])

        # Rule base — safety is given highest weight
        pq  = self._path_quality
        eq  = self._energy_quality
        sm  = self._smoothness
        sf  = self._safety
        fit = self._fitness

        rules = [
            # Safety = poor -> very_poor (collision risk = unacceptable)
            ctrl.Rule(sf["poor"], fit["very_poor"]),
            # Safety = acceptable + path poor + energy poor -> poor
            ctrl.Rule(sf["acceptable"] & pq["poor"] & eq["poor"], fit["poor"]),
            # Safety = acceptable + path acceptable -> acceptable
            ctrl.Rule(sf["acceptable"] & pq["acceptable"], fit["acceptable"]),
            # All acceptable -> acceptable
            ctrl.Rule(sf["acceptable"] & pq["acceptable"] &
                      eq["acceptable"] & sm["acceptable"], fit["acceptable"]),
            # Safety = good + path good -> good
            ctrl.Rule(sf["good"] & pq["good"], fit["good"]),
            # Safety = good + smoothness good -> good
            ctrl.Rule(sf["good"] & sm["good"], fit["good"]),
            # Safety = good + path good + energy good + smoothness good -> excellent
            ctrl.Rule(sf["good"] & pq["good"] & eq["good"] & sm["good"], fit["excellent"]),
            # Fallback: safety good but others poor -> acceptable
            ctrl.Rule(sf["good"] & pq["poor"], fit["acceptable"]),
        ]

        ctrl_system = ctrl.ControlSystem(rules)
        self._simulation = ctrl.ControlSystemSimulation(ctrl_system)

    def evaluate(self, normalized_objectives: Dict[str, float]) -> float:
        """Compute fitness from normalized objective qualities.

        Args:
            normalized_objectives: dict with keys
                "path_quality", "energy_quality", "smoothness", "safety"
                all in [0, 1] where 1.0 means best performance.

        Returns:
            fitness: float in [0, 1].
        """
        pq = float(np.clip(normalized_objectives.get("path_quality", 0.5), 0, 1))
        eq = float(np.clip(normalized_objectives.get("energy_quality", 0.5), 0, 1))
        sm = float(np.clip(normalized_objectives.get("smoothness", 0.5), 0, 1))
        sf = float(np.clip(normalized_objectives.get("safety", 0.5), 0, 1))

        if not self._use_skfuzzy:
            # Weighted average fallback: safety has highest weight
            return float(0.2 * pq + 0.15 * eq + 0.15 * sm + 0.5 * sf)

        try:
            sim = self._simulation
            sim.input["path_quality"]   = pq
            sim.input["energy_quality"] = eq
            sim.input["smoothness"]     = sm
            sim.input["safety"]         = sf
            sim.compute()
            return float(sim.output["fitness"])
        except Exception:
            # Fallback if FIS fails (e.g., no rule fires)
            return float(0.2 * pq + 0.15 * eq + 0.15 * sm + 0.5 * sf)

    def batch_evaluate(self,
                        solutions: List["Solution"],
                        objective_fns: List["ObjectiveFunction"],
                        obstacle_map=None) -> List[float]:
        """Normalize objectives across the population, then evaluate each solution.

        Args:
            solutions: list of Solution objects with pre-computed path results.
            objective_fns: ordered list [path_length, energy, smoothness, safety].
            obstacle_map: ObstacleMap for evaluation.

        Returns:
            List of fitness scores, one per solution.
        """
        if not solutions:
            return []

        # Compute raw objective values for all solutions
        raw: Dict[str, List[float]] = {fn.name: [] for fn in objective_fns}
        for sol in solutions:
            for fn in objective_fns:
                val = fn.evaluate(sol.path, obstacle_map)
                raw[fn.name].append(val)

        # Compute normalized qualities (1 = best) for each solution
        scores = []
        for i in range(len(solutions)):
            norm = {}
            for fn in objective_fns:
                pop = raw[fn.name]
                # normalized() returns a cost: 0=best, 1=worst (regardless of minimize flag)
                cost = fn.normalized(pop[i], pop)
                # Flip to quality: 1=best, 0=worst (for FIS antecedents)
                quality = 1.0 - cost
                if fn.name == "path_length":
                    norm["path_quality"] = quality
                elif fn.name == "energy":
                    norm["energy_quality"] = quality
                elif fn.name == "smoothness":
                    norm["smoothness"] = quality
                elif fn.name == "safety":
                    norm["safety"] = quality

            scores.append(self.evaluate(norm))

        return scores
