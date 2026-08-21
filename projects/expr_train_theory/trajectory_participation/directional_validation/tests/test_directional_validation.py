import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.trajectory_participation.directional_validation.analysis import (
    monotonicity_summary,
    rank_correlation,
    run_reference_study,
    save_results,
)


class DirectionalValidationTests(unittest.TestCase):
    def test_rank_correlation_increasing(self):
        self.assertAlmostEqual(rank_correlation([1, 2, 3], [2, 4, 8]), 1.0)

    def test_rank_correlation_decreasing(self):
        self.assertAlmostEqual(rank_correlation([1, 2, 3], [8, 4, 2]), -1.0)

    def test_monotonicity_counts_group_means(self):
        result = monotonicity_summary(
            [1, 1, 2, 2, 3, 3],
            [1.0, 1.2, 2.0, 2.2, 1.5, 1.7],
        )
        self.assertEqual(result["positive_steps"], 1)
        self.assertEqual(result["negative_steps"], 1)
        self.assertEqual(result["flat_steps"], 0)

    def test_reference_study_is_reproducible(self):
        a_rows, a_summary, a_paired = run_reference_study(
            layers=(1, 2), qubits=(1,), samples=2, seed=7
        )
        b_rows, b_summary, b_paired = run_reference_study(
            layers=(1, 2), qubits=(1,), samples=2, seed=7
        )
        self.assertEqual(a_rows, b_rows)
        self.assertEqual(a_paired, b_paired)
        self.assertEqual(len(a_summary), len(b_summary))
        for left, right in zip(a_summary, b_summary):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                if isinstance(left[key], float) and np.isnan(left[key]):
                    self.assertTrue(np.isnan(right[key]))
                else:
                    self.assertEqual(left[key], right[key])

    def test_reference_study_has_paired_reupload_rows(self):
        rows, summaries, paired = run_reference_study(
            layers=(1, 2), qubits=(1, 2), samples=2, seed=4
        )
        self.assertEqual(len(rows), 2 * 2 * 2 * 2)
        self.assertEqual(len(paired), 2 * 2 * 2)
        self.assertTrue(any(row["metric"] == "d_tp" for row in summaries))
        self.assertTrue(all(np.isfinite(row["d_tp_delta_reupload"]) for row in paired))

    def test_save_results(self):
        rows, summaries, paired = run_reference_study(
            layers=(1, 2), qubits=(1,), samples=1, seed=3
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_results(tmp, rows, summaries, paired, {"test": True})
            for name in (
                "directional_raw.csv",
                "directional_depth_summary.csv",
                "directional_reupload_pairs.csv",
                "directional_metadata.json",
            ):
                self.assertTrue((Path(tmp) / name).exists())


if __name__ == "__main__":
    unittest.main()
