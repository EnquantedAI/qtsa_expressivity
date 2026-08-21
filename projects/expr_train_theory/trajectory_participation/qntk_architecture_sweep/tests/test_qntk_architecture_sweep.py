import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.trajectory_participation.qntk_architecture_sweep.study import (
    SweepConfig,
    evaluate_config,
    iter_configs,
    metric_correlations,
    save_results,
    summarize_rows,
)


class QNTKArchitectureSweepTests(unittest.TestCase):
    def test_iter_configs(self):
        configs = list(iter_configs((1, 2), (1,), (None, "Y"), (False, True)))
        self.assertEqual(len(configs), 8)
        self.assertTrue(all(config.n_qubits == 1 for config in configs))

    def test_small_evaluation_is_finite(self):
        config = SweepConfig(1, 1, None, False)
        rows = evaluate_config(config, samples=1, dataset_size=3, seed=7)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(np.isfinite(row["d_tp_normalized_mean"]))
        self.assertTrue(np.isfinite(row["qntk_trace"]))
        self.assertGreaterEqual(row["qntk_rank"], 0)
        self.assertLessEqual(row["qntk_rank"], 3)

    def test_summary_groups_samples(self):
        config = SweepConfig(1, 1, "Y", False)
        rows = evaluate_config(config, samples=2, dataset_size=3, seed=8)
        summary = summarize_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["samples"], 2)

    def test_correlations_have_expected_pairs(self):
        rows = []
        for seed in range(4):
            rows.extend(
                evaluate_config(
                    SweepConfig(1 + seed % 2, 1, None, False),
                    samples=1,
                    dataset_size=3,
                    seed=20 + seed,
                )
            )
        correlations = metric_correlations(rows)
        self.assertEqual(len(correlations), 4)
        self.assertTrue(all(item["samples"] == len(rows) for item in correlations))

    def test_save_results(self):
        config = SweepConfig(1, 1, None, False)
        rows = evaluate_config(config, samples=1, dataset_size=3, seed=4)
        summary = summarize_rows(rows)
        correlations = metric_correlations(rows)
        with tempfile.TemporaryDirectory() as tmp:
            save_results(tmp, rows, summary, correlations, {"seed": 4})
            expected = (
                "qntk_architecture_raw.csv",
                "qntk_architecture_summary.csv",
                "qntk_architecture_correlations.csv",
                "qntk_architecture_metadata.json",
            )
            for name in expected:
                self.assertTrue((Path(tmp) / name).exists())


if __name__ == "__main__":
    unittest.main()
