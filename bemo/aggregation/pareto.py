"""Pareto optimality: non-dominated sorting and Pareto front management."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bemo.objectives.base import PathResult


@dataclass
class Solution:
    """A candidate solution with its multi-objective evaluation."""
    solution_id: str
    algorithm: str
    path: "PathResult"
    # objectives: shape (4,) — [path_length, energy, smoothness, safety_negated]
    # All in minimize-convention: lower is better.
    objectives: np.ndarray
    objective_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.objectives = np.asarray(self.objectives, dtype=float)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a dominates b: a <= b in all objectives, strictly < in at least one."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(solutions: List[Solution]) -> List[List[Solution]]:
    """NSGA-II style non-dominated sorting.

    Returns:
        List of Pareto fronts; fronts[0] is the Pareto-optimal set.
    """
    n = len(solutions)
    if n == 0:
        return []

    domination_count = np.zeros(n, dtype=int)
    dominated_sets: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(solutions[i].objectives, solutions[j].objectives):
                dominated_sets[i].append(j)
            elif dominates(solutions[j].objectives, solutions[i].objectives):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front: List[int] = []
        for i in fronts[current_front]:
            for j in dominated_sets[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    # Convert indices to solutions, drop the trailing empty front
    result = [[solutions[i] for i in front] for front in fronts if front]
    return result


class ParetoFront:
    """Maintains and updates the Pareto-optimal set incrementally."""

    def __init__(self, objective_names: Optional[List[str]] = None):
        self.objective_names = objective_names or [
            "path_length", "energy", "smoothness", "safety_neg"
        ]
        self._solutions: List[Solution] = []

    def add(self, solution: Solution) -> bool:
        """Add solution if it is non-dominated. Removes any it dominates.

        Returns:
            True if the solution was added to the front.
        """
        new_obj = solution.objectives

        # Check if dominated by any existing front member
        for existing in self._solutions:
            if dominates(existing.objectives, new_obj):
                return False  # new solution is dominated

        # Remove existing solutions dominated by new solution
        self._solutions = [
            s for s in self._solutions
            if not dominates(new_obj, s.objectives)
        ]
        self._solutions.append(solution)
        return True

    def get_front(self) -> List[Solution]:
        return list(self._solutions)

    def size(self) -> int:
        return len(self._solutions)

    def hypervolume(self, reference_point: Optional[np.ndarray] = None) -> float:
        """Compute dominated hypervolume (WFG algorithm for up to 4D).

        Uses a simple Monte Carlo approximation for simplicity.
        """
        if not self._solutions:
            return 0.0

        ref = reference_point
        if ref is None:
            ref = np.ones(len(self.objective_names)) * 1.1

        objectives = np.array([s.objectives for s in self._solutions])

        # Monte Carlo approximation
        n_samples = 10000
        rng = np.random.default_rng(0)
        lb = np.zeros(len(ref))
        samples = rng.uniform(lb, ref, size=(n_samples, len(ref)))

        dominated = np.zeros(n_samples, dtype=bool)
        for obj in objectives:
            dominated |= np.all(samples >= obj, axis=1)

        vol_total = np.prod(ref - lb)
        return float(dominated.mean() * vol_total)

    def to_array(self) -> np.ndarray:
        """Return objectives matrix, shape (N, M)."""
        if not self._solutions:
            return np.empty((0, len(self.objective_names)))
        return np.array([s.objectives for s in self._solutions])

    def __len__(self) -> int:
        return len(self._solutions)

    def __repr__(self) -> str:
        return f"ParetoFront(size={len(self._solutions)})"
