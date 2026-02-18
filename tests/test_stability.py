"""
Unit tests for stability computation functions.

Tests the core statistical claims:
  - Categorical stability (agreement ratio, entropy, modal label)
  - Continuous stability (mean, median, stdev, range, binning)
  - Krippendorff alpha (nominal and interval)
  - Fleiss kappa
  - Bootstrap CI
"""

import sys
from unittest.mock import MagicMock

# Mock psycopg2 before importing stability (avoids DB dependency in tests)
sys.modules['psycopg2'] = MagicMock()
sys.modules['psycopg2.extras'] = MagicMock()

import pytest
import numpy as np

from stability import (
    compute_categorical_stability,
    compute_continuous_stability,
    krippendorff_alpha_nominal,
    krippendorff_alpha_interval,
    fleiss_kappa,
    bootstrap_ci,
    bootstrap_stability_rate,
)


# ── Categorical Stability ─────────────────────────────────────────────

class TestCategoricalStability:
    def test_unanimous_agreement(self):
        labels = ['past', 'past', 'past', 'past', 'past']
        result = compute_categorical_stability(labels, threshold=0.8)
        assert result['agreement_ratio'] == 1.0
        assert result['modal_label'] == 'past'
        assert result['modal_count'] == 5
        assert result['is_stable'] is True
        assert result['label_entropy'] == 0.0

    def test_four_of_five_agreement(self):
        labels = ['past', 'past', 'past', 'past', 'present']
        result = compute_categorical_stability(labels, threshold=0.8)
        assert result['agreement_ratio'] == 0.8
        assert result['modal_label'] == 'past'
        assert result['is_stable'] is True  # 0.8 >= 0.8

    def test_three_of_five_unstable(self):
        labels = ['past', 'past', 'past', 'present', 'future']
        result = compute_categorical_stability(labels, threshold=0.8)
        assert result['agreement_ratio'] == 0.6
        assert result['modal_label'] == 'past'
        assert result['is_stable'] is False  # 0.6 < 0.8

    def test_even_split(self):
        labels = ['active', 'passive', 'active', 'passive']
        result = compute_categorical_stability(labels, threshold=0.8)
        assert result['agreement_ratio'] == 0.5
        assert result['is_stable'] is False

    def test_entropy_increases_with_disagreement(self):
        unanimous = compute_categorical_stability(['a', 'a', 'a', 'a', 'a'], 0.8)
        split = compute_categorical_stability(['a', 'a', 'b', 'b', 'c'], 0.8)
        assert split['label_entropy'] > unanimous['label_entropy']

    def test_two_labels(self):
        labels = ['present', 'absent']
        result = compute_categorical_stability(labels, threshold=0.8)
        assert result['agreement_ratio'] == 0.5
        assert result['is_stable'] is False


# ── Continuous Stability ──────────────────────────────────────────────

class TestContinuousStability:
    BINS = {'low': (0.0, 0.29), 'moderate': (0.3, 0.69), 'high': (0.7, 1.0)}

    def test_tight_cluster_stable(self):
        values = [0.5, 0.52, 0.48, 0.51, 0.49]
        result = compute_continuous_stability(values, self.BINS, max_range=0.2, max_stdev=0.10)
        assert result['is_stable'] is True
        assert result['modal_bin'] == 'moderate'
        assert 0.48 <= result['mean_value'] <= 0.52

    def test_spread_values_unstable(self):
        values = [0.1, 0.5, 0.9, 0.3, 0.7]
        result = compute_continuous_stability(values, self.BINS, max_range=0.2, max_stdev=0.10)
        assert result['is_stable'] is False
        assert result['range_value'] == 0.8

    def test_all_same_value(self):
        values = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = compute_continuous_stability(values, self.BINS, max_range=0.2, max_stdev=0.10)
        assert result['is_stable'] is True
        assert result['stdev_value'] == 0.0
        assert result['range_value'] == 0.0

    def test_boundary_values(self):
        values = [0.29, 0.30, 0.29, 0.30, 0.29]
        result = compute_continuous_stability(values, self.BINS, max_range=0.2, max_stdev=0.10)
        assert result['is_stable'] is True
        assert result['range_value'] <= 0.02

    def test_high_bin_assignment(self):
        values = [0.8, 0.85, 0.9, 0.75, 0.82]
        result = compute_continuous_stability(values, self.BINS, max_range=0.2, max_stdev=0.10)
        assert result['modal_bin'] == 'high'

    def test_bin_agreement_ratio(self):
        # All in same bin
        values = [0.1, 0.15, 0.2, 0.1, 0.12]
        result = compute_continuous_stability(values, self.BINS, max_range=0.2, max_stdev=0.10)
        assert result['bin_agreement_ratio'] == 1.0


# ── Krippendorff Alpha (Nominal) ─────────────────────────────────────

class TestKrippendorffNominal:
    def test_perfect_agreement(self):
        data = [
            ['a', 'a', 'a'],
            ['b', 'b', 'b'],
            ['a', 'a', 'a'],
        ]
        alpha = krippendorff_alpha_nominal(data)
        assert alpha == 1.0

    def test_all_same_label_trivial(self):
        data = [
            ['a', 'a', 'a'],
            ['a', 'a', 'a'],
        ]
        alpha = krippendorff_alpha_nominal(data)
        assert alpha == 1.0  # Trivial case

    def test_random_disagreement_near_zero(self):
        # Maximally random pattern
        np.random.seed(42)
        labels = ['a', 'b', 'c']
        data = [[labels[np.random.randint(3)] for _ in range(5)] for _ in range(100)]
        alpha = krippendorff_alpha_nominal(data)
        assert -0.15 <= alpha <= 0.15  # Should be near 0

    def test_complete_disagreement(self):
        # Systematic disagreement
        data = [
            ['a', 'b'],
            ['b', 'a'],
            ['a', 'b'],
            ['b', 'a'],
        ]
        alpha = krippendorff_alpha_nominal(data)
        assert alpha < 0  # Systematic disagreement gives negative alpha

    def test_with_missing_data(self):
        data = [
            ['a', 'a', None],
            ['b', 'b', 'b'],
            [None, 'a', 'a'],
        ]
        alpha = krippendorff_alpha_nominal(data)
        assert alpha == 1.0  # All present data agrees

    def test_single_item_returns_zero(self):
        data = [['a', 'b']]
        alpha = krippendorff_alpha_nominal(data)
        # With only one item, alpha should handle gracefully
        assert isinstance(alpha, float)


# ── Krippendorff Alpha (Interval) ────────────────────────────────────

class TestKrippendorffInterval:
    def test_perfect_agreement(self):
        data = [
            [0.5, 0.5, 0.5],
            [0.8, 0.8, 0.8],
            [0.2, 0.2, 0.2],
        ]
        alpha = krippendorff_alpha_interval(data)
        assert alpha == 1.0

    def test_high_agreement(self):
        data = [
            [0.50, 0.51, 0.49],
            [0.80, 0.79, 0.81],
            [0.20, 0.21, 0.19],
        ]
        alpha = krippendorff_alpha_interval(data)
        assert alpha > 0.9

    def test_random_values_low_alpha(self):
        np.random.seed(42)
        data = [[np.random.random() for _ in range(5)] for _ in range(50)]
        alpha = krippendorff_alpha_interval(data)
        assert alpha < 0.2

    def test_with_missing_data(self):
        data = [
            [0.5, 0.5, None],
            [0.8, None, 0.8],
            [None, 0.2, 0.2],
        ]
        alpha = krippendorff_alpha_interval(data)
        assert alpha == 1.0


# ── Fleiss Kappa ─────────────────────────────────────────────────────

class TestFleissKappa:
    def test_perfect_agreement(self):
        data = [
            ['a', 'a', 'a'],
            ['b', 'b', 'b'],
            ['c', 'c', 'c'],
        ]
        kappa = fleiss_kappa(data)
        assert kappa == 1.0

    def test_no_agreement(self):
        # Each item gets uniform spread of labels
        data = [
            ['a', 'b', 'c'],
            ['a', 'b', 'c'],
            ['a', 'b', 'c'],
        ]
        kappa = fleiss_kappa(data)
        assert kappa < 0.1  # Near chance agreement

    def test_moderate_agreement(self):
        data = [
            ['a', 'a', 'b'],
            ['b', 'b', 'a'],
            ['a', 'a', 'a'],
            ['b', 'b', 'b'],
        ]
        kappa = fleiss_kappa(data)
        assert 0.0 < kappa < 1.0

    def test_with_missing_data(self):
        data = [
            ['a', 'a', None],
            ['b', 'b', 'b'],
        ]
        kappa = fleiss_kappa(data)
        assert isinstance(kappa, float)


# ── Bootstrap CI ─────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_stability_rate_all_stable(self):
        flags = [True] * 100
        point, lo, hi = bootstrap_stability_rate(flags)
        assert point == 1.0
        assert lo == 1.0
        assert hi == 1.0

    def test_stability_rate_none_stable(self):
        flags = [False] * 100
        point, lo, hi = bootstrap_stability_rate(flags)
        assert point == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_stability_rate_mixed(self):
        flags = [True] * 70 + [False] * 30
        point, lo, hi = bootstrap_stability_rate(flags)
        assert 0.65 <= point <= 0.75
        assert lo < point
        assert hi > point
        assert lo > 0.5  # CI should be well above 50%

    def test_bootstrap_ci_returns_three_floats(self):
        def mean_func(data):
            return sum(data) / len(data)
        point, lo, hi = bootstrap_ci(mean_func, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(point, float)
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo <= point <= hi
