import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.trajectory_participation.architecture_sweep.sweep import (
    SweepConfig,
    evaluate_config,
    iter_configs,
    run_sweep,
    save_sweep,
    summarize_rows,
)


def fake_snapshots(inputs, weights, *, n_qubits, fm_style, reup_style):
    dim = 2**n_qubits
    n_snapshots = weights.shape[0] + 1
    states = np.zeros((n_snapshots, dim), dtype=complex)
    for index in range(n_snapshots):
        states[index, index % dim] = 1.0
    return states


class ArchitectureSweepTests(unittest.TestCase):
    def test_config_grid(self):
        configs = list(iter_configs([1, 2], [1, 3], ["Y"], [None, "X"]))
        self.assertEqual(len(configs), 8)
        self.assertEqual(configs[0], SweepConfig(1, 1, "Y", None))

    def test_evaluate_config_uses_expected_bounds(self):
        config = SweepConfig(3, 2, "Y", None)
        rows = evaluate_config(
            config,
            samples=2,
            input_size=5,
            snapshot_fn=fake_snapshots,
        )
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["d_tp"], 4.0)
        self.assertAlmostEqual(rows[0]["d_tp_normalized"], 1.0)
        self.assertEqual(rows[0]["numerical_rank"], 4)

    def test_summary_groups_rows(self):
        config = SweepConfig(2, 2, "Y", "X")
        rows = evaluate_config(
            config,
            samples=3,
            snapshot_fn=fake_snapshots,
        )
        summary = summarize_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["samples"], 3)
        self.assertAlmostEqual(summary[0]["d_tp_std"], 0.0)

    def test_run_sweep_is_reproducible(self):
        kwargs = dict(
            layers=(1, 2),
            qubits=(1, 2),
            feature_maps=("Y",),
            reupload_styles=(None,),
            samples=2,
            seed=11,
            snapshot_fn=fake_snapshots,
        )
        first, first_summary = run_sweep(**kwargs)
        second, second_summary = run_sweep(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)

    def test_save_sweep(self):
        config = SweepConfig(1, 1, "Y", None)
        rows = evaluate_config(config, samples=1, snapshot_fn=fake_snapshots)
        summary = summarize_rows(rows)
        with tempfile.TemporaryDirectory() as tmp:
            save_sweep(tmp, rows, summary, {"seed": 1})
            self.assertTrue((Path(tmp) / "trajectory_sweep_raw.csv").exists())
            self.assertTrue((Path(tmp) / "trajectory_sweep_summary.csv").exists())
            self.assertTrue((Path(tmp) / "trajectory_sweep_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
