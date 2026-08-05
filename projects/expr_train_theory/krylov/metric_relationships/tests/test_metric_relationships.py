import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.krylov.circuit_ensemble.study import EnsembleConfig
from projects.expr_train_theory.krylov.metric_relationships.analysis import (
    CorrelationConfig,
    correlation_matrix,
    run_correlation_study,
)


class MetricRelationshipTests(unittest.TestCase):
    def test_perfect_linear_relation(self):
        values = np.column_stack((np.arange(6.0), 3.0 * np.arange(6.0) + 2.0))
        row = correlation_matrix(values, ("x", "y"))[0]
        self.assertAlmostEqual(row["pearson"], 1.0, places=12)
        self.assertAlmostEqual(row["spearman"], 1.0, places=12)

    def test_monotone_nonlinear_relation(self):
        x = np.arange(1.0, 8.0)
        values = np.column_stack((x, x**2))
        row = correlation_matrix(values, ("x", "y"))[0]
        self.assertLess(row["pearson"], 1.0)
        self.assertAlmostEqual(row["spearman"], 1.0, places=12)

    def test_constant_column_returns_nan(self):
        values = np.column_stack((np.ones(5), np.arange(5.0)))
        row = correlation_matrix(values, ("x", "y"))[0]
        self.assertTrue(np.isnan(row["pearson"]))
        self.assertTrue(np.isnan(row["spearman"]))

    def test_unknown_metric_is_rejected(self):
        config = CorrelationConfig(
            ensemble=EnsembleConfig(2, (1,), ("none",), samples=2),
            metrics=("layer_orbit_fraction", "missing"),
        )
        with self.assertRaises(ValueError):
            config.validate()

    def test_study_is_reproducible(self):
        config = CorrelationConfig(
            ensemble=EnsembleConfig(2, (1, 2), ("none", "linear"), samples=4, seed=17),
            metrics=("layer_orbit_fraction", "final_mean_single_qubit_entropy"),
        )
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run_correlation_study(config, first)
            b = run_correlation_study(config, second)
        self.assertEqual(a[0], b[0])
        for rows_a, rows_b in zip(a[1:], b[1:]):
            self.assertEqual(len(rows_a), len(rows_b))
            for row_a, row_b in zip(rows_a, rows_b):
                self.assertEqual(set(row_a), set(row_b))
                for key in row_a:
                    first = row_a[key]
                    second = row_b[key]
                    if isinstance(first, float) and np.isnan(first):
                        self.assertTrue(np.isnan(second))
                    else:
                        self.assertEqual(first, second)

    def test_result_files_are_written(self):
        config = CorrelationConfig(
            ensemble=EnsembleConfig(2, (1,), ("none", "linear"), samples=3, seed=8),
            metrics=("layer_orbit_fraction", "final_mean_single_qubit_entropy"),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_correlation_study(config, output)
            self.assertTrue((output / "metric_correlations_global.csv").exists())
            self.assertTrue((output / "metric_correlations_grouped.csv").exists())
            self.assertTrue((output / "metric_correlations_metadata.json").exists())
            self.assertTrue((output / "ensemble" / "circuit_ensemble_raw.csv").exists())


if __name__ == "__main__":
    unittest.main()
