import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.trajectory_participation.shared_qnn_cross_metric.study import (
    SweepConfig,
    evaluate_config,
    evaluate_trajectory,
    metric_correlations,
    save_results,
    summarize_rows,
)


def _ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _toy_trajectory(theta):
    state = np.array([1.0, 0.0], dtype=np.complex128)
    snapshots = [state.copy()]
    for value in np.asarray(theta, dtype=float).reshape(-1):
        state = _ry(value) @ state
        snapshots.append(state.copy())
    return np.asarray(snapshots)


def _fake_snapshot_fn(inputs, weights, *, n_qubits, fm_style, reup_style):
    if n_qubits != 1:
        raise ValueError("test helper uses one qubit")
    theta = np.asarray(weights, dtype=float)[:, 0, 1]
    return _toy_trajectory(theta)


class CrossMetricSweepTests(unittest.TestCase):
    def test_reference_trajectory_detects_parameter_redundancy(self):
        result = evaluate_trajectory(np.array([0.3, -0.5]), _toy_trajectory)
        self.assertEqual(result["parameter_count"], 2)
        self.assertEqual(result["trajectory_rank"], 2)
        self.assertEqual(result["qfim_rank"], 1)
        self.assertEqual(result["cfim_rank"], 1)

    def test_evaluate_config_accepts_injected_snapshot_function(self):
        config = SweepConfig(2, 1, "Y", None)
        rows = evaluate_config(
            config,
            samples=2,
            input_size=1,
            seed=10,
            snapshot_fn=_fake_snapshot_fn,
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["qfim_rank"] == 1 for row in rows))

    def test_summary_groups_architectures(self):
        rows = [
            {
                "n_layers": 1, "n_qubits": 1, "fm_style": "Y", "reup_style": "none",
                "d_tp": 1.2, "d_tp_rank_fraction": 0.6, "d_tp_ceiling_fraction": 0.6,
                "trajectory_rank": 2, "qfim_relative_rank": 1.0, "qfim_trace": 1.0,
                "cfim_relative_rank": 1.0, "cfim_trace": 1.0, "rank_gap_qfim_cfim": 0,
            },
            {
                "n_layers": 1, "n_qubits": 1, "fm_style": "Y", "reup_style": "none",
                "d_tp": 1.4, "d_tp_rank_fraction": 0.7, "d_tp_ceiling_fraction": 0.7,
                "trajectory_rank": 2, "qfim_relative_rank": 1.0, "qfim_trace": 1.0,
                "cfim_relative_rank": 1.0, "cfim_trace": 1.0, "rank_gap_qfim_cfim": 0,
            },
        ]
        summary = summarize_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["d_tp_mean"], 1.3)

    def test_correlations_return_requested_pairs(self):
        rows = []
        for index in range(4):
            rows.append({
                "d_tp": float(index + 1),
                "qfim_relative_rank": float(index + 1),
                "qfim_trace": float(index + 2),
                "cfim_relative_rank": float(index + 1),
                "cfim_trace": float(index + 3),
            })
        correlations = metric_correlations(rows)
        self.assertEqual(len(correlations), 6)
        self.assertAlmostEqual(correlations[0]["spearman"], 1.0)

    def test_save_results_writes_all_outputs(self):
        rows = [{"a": 1}]
        summary = [{"b": 2}]
        correlations = [{"c": 3}]
        with tempfile.TemporaryDirectory() as tmp:
            save_results(tmp, rows, summary, correlations, {"seed": 1})
            names = {path.name for path in Path(tmp).iterdir()}
        self.assertIn("shared_qnn_cross_metric_raw.csv", names)
        self.assertIn("shared_qnn_cross_metric_summary.csv", names)
        self.assertIn("shared_qnn_cross_metric_correlations.csv", names)
        self.assertIn("shared_qnn_cross_metric_metadata.json", names)


if __name__ == "__main__":
    unittest.main()
