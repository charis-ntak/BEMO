"""Tests for Pareto front computation."""
import numpy as np
import pytest

from bemo.aggregation.pareto import (Solution, ParetoFront, dominates,
                                       fast_non_dominated_sort)
from tests.fixtures import make_straight_path


def make_solution(obj_vec, sol_id="test", algorithm="test"):
    return Solution(
        solution_id=sol_id,
        algorithm=algorithm,
        path=make_straight_path(),
        objectives=np.array(obj_vec, dtype=float),
    )


class TestDominates:
    def test_clear_dominance(self):
        a = np.array([1.0, 1.0, 1.0, 1.0])
        b = np.array([2.0, 2.0, 2.0, 2.0])
        assert dominates(a, b)
        assert not dominates(b, a)

    def test_equal_not_dominates(self):
        a = np.array([1.0, 1.0])
        assert not dominates(a, a)

    def test_partial_dominance(self):
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 1.0])
        assert not dominates(a, b)
        assert not dominates(b, a)

    def test_one_objective_better(self):
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 2.0])
        assert dominates(a, b)


class TestFastNonDominatedSort:
    def test_empty(self):
        assert fast_non_dominated_sort([]) == []

    def test_single_solution(self):
        sol = make_solution([1, 2, 3, 4])
        fronts = fast_non_dominated_sort([sol])
        assert len(fronts) == 1
        assert len(fronts[0]) == 1

    def test_two_nondominated(self):
        a = make_solution([1.0, 2.0], "a")
        b = make_solution([2.0, 1.0], "b")
        fronts = fast_non_dominated_sort([a, b])
        assert len(fronts) == 1
        assert len(fronts[0]) == 2

    def test_one_dominates(self):
        a = make_solution([1.0, 1.0], "a")
        b = make_solution([2.0, 2.0], "b")
        fronts = fast_non_dominated_sort([a, b])
        assert len(fronts) == 2
        assert fronts[0][0].solution_id == "a"
        assert fronts[1][0].solution_id == "b"

    def test_multiple_fronts(self):
        # a=[1,1] dominates b, c, d. d=[1,2] dominates b, c. b=[2,2] dominates c.
        sols = [
            make_solution([1, 1], "a"),  # front 0 only
            make_solution([2, 2], "b"),  # front 1 (dominated by a, d)
            make_solution([3, 3], "c"),  # front 2 (dominated by a, b, d)
            make_solution([1, 2], "d"),  # front 1 (dominated by a)
        ]
        fronts = fast_non_dominated_sort(sols)
        assert len(fronts) >= 2
        front0_ids = {s.solution_id for s in fronts[0]}
        assert "a" in front0_ids
        # "c" should never be in front 0
        assert "c" not in front0_ids


class TestParetoFront:
    def test_add_first_solution(self):
        pf = ParetoFront()
        sol = make_solution([1, 2, 3, 4])
        added = pf.add(sol)
        assert added
        assert pf.size() == 1

    def test_add_dominated_rejected(self):
        pf = ParetoFront()
        good = make_solution([1, 1, 1, 1], "good")
        bad = make_solution([2, 2, 2, 2], "bad")
        pf.add(good)
        added = pf.add(bad)
        assert not added
        assert pf.size() == 1

    def test_add_dominating_removes_dominated(self):
        pf = ParetoFront()
        bad = make_solution([2, 2, 2, 2], "bad")
        good = make_solution([1, 1, 1, 1], "good")
        pf.add(bad)
        pf.add(good)
        assert pf.size() == 1
        assert pf.get_front()[0].solution_id == "good"

    def test_nondominated_solutions_coexist(self):
        pf = ParetoFront()
        a = make_solution([1, 2], "a")
        b = make_solution([2, 1], "b")
        pf.add(a)
        pf.add(b)
        assert pf.size() == 2

    def test_hypervolume_positive(self):
        pf = ParetoFront()
        pf.add(make_solution([0.2, 0.3, 0.1, 0.4]))
        hv = pf.hypervolume()
        assert hv > 0

    def test_hypervolume_empty(self):
        pf = ParetoFront()
        assert pf.hypervolume() == 0.0

    def test_to_array(self):
        pf = ParetoFront()
        pf.add(make_solution([1, 2, 3, 4]))
        arr = pf.to_array()
        assert arr.shape == (1, 4)
