"""Tests for Mamdani FIS — including corner case rule completeness."""
import numpy as np
import pytest

from bemo.aggregation.fuzzy_fis import MamdaniFIS, SKFUZZY_AVAILABLE


@pytest.fixture
def fis():
    return MamdaniFIS()


class TestMamdaniFIS:
    def test_init(self, fis):
        assert fis is not None

    def test_output_in_range(self, fis):
        result = fis.evaluate({
            "path_quality": 0.5,
            "energy_quality": 0.5,
            "smoothness": 0.5,
            "safety": 0.5,
        })
        assert 0.0 <= result <= 1.0

    def test_all_worst(self, fis):
        """Corner case: all objectives at worst (0) — should not produce NaN."""
        result = fis.evaluate({
            "path_quality": 0.0,
            "energy_quality": 0.0,
            "smoothness": 0.0,
            "safety": 0.0,
        })
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0
        assert result < 0.5  # should be a poor score

    def test_all_best(self, fis):
        """Corner case: all objectives at best (1) — should not produce NaN."""
        result = fis.evaluate({
            "path_quality": 1.0,
            "energy_quality": 1.0,
            "smoothness": 1.0,
            "safety": 1.0,
        })
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0
        assert result > 0.5  # should be a good score

    def test_safety_dominates(self, fis):
        """Low safety should result in low fitness regardless of other objectives."""
        low_safety = fis.evaluate({
            "path_quality": 1.0,
            "energy_quality": 1.0,
            "smoothness": 1.0,
            "safety": 0.0,
        })
        high_safety = fis.evaluate({
            "path_quality": 0.5,
            "energy_quality": 0.5,
            "smoothness": 0.5,
            "safety": 1.0,
        })
        assert low_safety < high_safety

    def test_clipping(self, fis):
        """Values outside [0, 1] should be clipped, not crash."""
        result = fis.evaluate({
            "path_quality": 1.5,
            "energy_quality": -0.2,
            "smoothness": 0.5,
            "safety": 0.8,
        })
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0

    def test_monotone_safety(self, fis):
        """Increasing safety quality should increase fitness (all else equal)."""
        scores = [
            fis.evaluate({
                "path_quality": 0.5,
                "energy_quality": 0.5,
                "smoothness": 0.5,
                "safety": s,
            })
            for s in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        # Should be non-decreasing
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 0.05  # small tolerance for FIS

    def test_batch_evaluate_returns_correct_count(self, fis):
        from tests.fixtures import make_straight_path
        from bemo.aggregation.pareto import Solution
        from bemo.objectives import (PathLengthObjective, EnergyObjective,
                                      SmoothnessObjective, SafetyObjective)

        objectives = [PathLengthObjective(), EnergyObjective(),
                      SmoothnessObjective(), SafetyObjective()]
        solutions = [
            Solution("s1", "RRT*", make_straight_path(), np.array([5, 1, 0.1, -2])),
            Solution("s2", "APF", make_straight_path(), np.array([7, 2, 0.2, -1])),
        ]
        scores = fis.batch_evaluate(solutions, objectives, None)
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert all(not np.isnan(s) for s in scores)
