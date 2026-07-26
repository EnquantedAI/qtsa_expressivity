"""Tests for sweep aggregation."""

import math
import unittest

from projects.expr_train_theory.qfim.experiments.architecture_sweep import (
    ArchitectureSweepConfig,
    aggregate_architecture_rows,
)


class ArchitectureSweepAggregationTests(unittest.TestCase):
    def test_configuration_rejects_nonpositive_sample_count(self):
        config = ArchitectureSweepConfig(samples_per_architecture=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_rows_are_aggregated_by_architecture(self):
        rows = [
            {
                "width": 2,
                "depth": 1,
                "feature_map": "zzfm",
                "parameter_count": 6,
                "numerical_rank": 4,
                "relative_rank": 4 / 6,
                "trace": 10.0,
                "minimum_eigenvalue": -1e-12,
                "positive_condition_number": 5.0,
            },
            {
                "width": 2,
                "depth": 1,
                "feature_map": "zzfm",
                "parameter_count": 6,
                "numerical_rank": 6,
                "relative_rank": 1.0,
                "trace": 14.0,
                "minimum_eigenvalue": 0.0,
                "positive_condition_number": math.inf,
            },
        ]

        summaries = aggregate_architecture_rows(rows)

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["parameter_count"], 6)
        self.assertAlmostEqual(summary["mean_rank"], 5.0)
        self.assertAlmostEqual(summary["std_rank"], 1.0)
        self.assertAlmostEqual(summary["mean_trace"], 12.0)
        self.assertAlmostEqual(summary["mean_positive_condition_number"], 5.0)


if __name__ == "__main__":
    unittest.main()
