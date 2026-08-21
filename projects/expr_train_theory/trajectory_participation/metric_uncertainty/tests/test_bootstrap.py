import tempfile
import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation.metric_uncertainty.bootstrap import (
    bootstrap_correlation,
    bootstrap_mean,
    compare_metric_pairs,
    save_report,
)


class MetricUncertaintyTests(unittest.TestCase):
    def test_constant_mean_has_zero_width_interval(self):
        result = bootstrap_mean([2.0, 2.0, 2.0], bootstrap_samples=100, seed=1)
        self.assertAlmostEqual(result.estimate, 2.0)
        self.assertAlmostEqual(result.lower, 2.0)
        self.assertAlmostEqual(result.upper, 2.0)

    def test_perfect_positive_correlation(self):
        x = np.arange(1.0, 8.0)
        result = bootstrap_correlation(x, 3.0 * x + 2.0, bootstrap_samples=200, seed=2)
        self.assertAlmostEqual(result.estimate, 1.0)
        self.assertGreater(result.lower, 0.99)

    def test_perfect_negative_spearman(self):
        x = np.arange(1.0, 8.0)
        result = bootstrap_correlation(
            x, -x, method="spearman", bootstrap_samples=200, seed=3
        )
        self.assertAlmostEqual(result.estimate, -1.0)
        self.assertLess(result.upper, -0.99)

    def test_pair_report_contains_both_methods(self):
        rows = [
            {"a": 1.0, "b": 2.0},
            {"a": 2.0, "b": 3.0},
            {"a": 3.0, "b": 5.0},
            {"a": 4.0, "b": 8.0},
        ]
        report = compare_metric_pairs(rows, [("a", "b")], bootstrap_samples=100)
        self.assertEqual({row["method"] for row in report}, {"pearson", "spearman"})

    def test_report_can_be_saved(self):
        rows = [{"left": "a", "right": "b", "method": "pearson", "estimate": 0.5,
                 "lower": 0.1, "upper": 0.8, "n": 10, "bootstrap_samples": 100}]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, json_path = save_report(tmp, rows, {"seed": 1})
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())


if __name__ == "__main__":
    unittest.main()
