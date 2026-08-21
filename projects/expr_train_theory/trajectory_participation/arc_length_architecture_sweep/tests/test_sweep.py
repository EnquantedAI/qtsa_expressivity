import unittest

import numpy as np

from ...architecture_sweep.sweep import SweepConfig
from ...sampling import qubit_arc_states
from ..sweep import compare_config, run_comparison_sweep, summarize_rows


def arc_snapshot_fn(inputs, weights, *, n_qubits, fm_style, reup_style):
    del inputs, fm_style, reup_style
    n_layers = weights.shape[0]
    parameters = np.linspace(0.0, np.pi / 2.0, n_layers + 1)
    states = qubit_arc_states(parameters)
    if n_qubits == 1:
        return states

    padded = np.zeros((states.shape[0], 2**n_qubits), dtype=complex)
    padded[:, :2] = states
    return padded


class ArcLengthArchitectureSweepTests(unittest.TestCase):
    def test_compare_config_returns_matched_metrics(self):
        config = SweepConfig(4, 1, "Y", None)
        rows = compare_config(config, samples=3, snapshot_fn=arc_snapshot_fn)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row["n_snapshots"], 5)
            self.assertGreaterEqual(row["d_tp_equal"], 1.0)
            self.assertGreaterEqual(row["d_tp_arc"], 1.0)
            self.assertAlmostEqual(row["path_length_fs"], np.pi / 2.0, places=10)

    def test_summary_groups_samples(self):
        config = SweepConfig(3, 1, "Y", None)
        rows = compare_config(config, samples=4, snapshot_fn=arc_snapshot_fn)
        summary = summarize_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["samples"], 4)

    def test_same_geometric_arc_has_more_stable_arc_weighted_value(self):
        shallow = compare_config(
            SweepConfig(2, 1, "Y", None), samples=1, snapshot_fn=arc_snapshot_fn
        )[0]
        deep = compare_config(
            SweepConfig(16, 1, "Y", None), samples=1, snapshot_fn=arc_snapshot_fn
        )[0]
        equal_gap = abs(shallow["d_tp_equal"] - deep["d_tp_equal"])
        arc_gap = abs(shallow["d_tp_arc"] - deep["d_tp_arc"])
        self.assertLess(arc_gap, equal_gap)

    def test_run_sweep_keeps_all_requested_configurations(self):
        rows, summary = run_comparison_sweep(
            layers=(1, 2),
            qubits=(1,),
            feature_maps=("Y",),
            reupload_styles=(None, "Y"),
            samples=2,
            snapshot_fn=arc_snapshot_fn,
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(summary), 4)

    def test_invalid_sample_count_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_config(
                SweepConfig(2, 1, "Y", None),
                samples=0,
                snapshot_fn=arc_snapshot_fn,
            )


if __name__ == "__main__":
    unittest.main()
