import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.trajectory_participation.graph_topology.cross_metric import (
    save_cross_metric_results,
    summarize_cross_metric,
    topology_cross_metric_study,
    topology_final_state,
    z0_expectation,
)
from projects.expr_train_theory.trajectory_participation.graph_topology.study import (
    topology_snapshots,
)


class GraphTopologyCrossMetricTests(unittest.TestCase):
    def test_final_state_matches_last_snapshot(self):
        features = np.array([0.2, -0.4])
        parameters = np.array([[[0.1, 0.2, 0.3], [-0.2, 0.4, -0.1]]])
        edges = ((0, 1),)
        snapshots = topology_snapshots(
            features, parameters, n_qubits=2, edges=edges
        )
        final = topology_final_state(
            features, parameters, n_qubits=2, edges=edges
        )
        np.testing.assert_allclose(final, snapshots[-1], atol=1e-12)

    def test_z0_expectation_is_bounded(self):
        features = np.array([0.6, -0.2])
        parameters = np.array([[[0.3, 0.1, -0.2], [0.4, -0.5, 0.2]]])
        value = z0_expectation(
            features, parameters, n_qubits=2, edges=((0, 1),)
        )
        self.assertLessEqual(abs(value), 1.0 + 1e-12)

    def test_matched_study_records_all_three_metric_families(self):
        rows = topology_cross_metric_study(
            n_qubits=2,
            n_layers=1,
            topologies=("none", "line"),
            parameter_samples=1,
            data_points=3,
            seed=11,
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIn("d_tp_mean", row)
            self.assertIn("qfim_rank_mean", row)
            self.assertIn("qfim_trace_mean", row)
            self.assertIn("qntk_rank", row)
            self.assertIn("qntk_effective_rank", row)
            self.assertTrue(np.isfinite(row["d_tp_mean"]))
            self.assertTrue(np.isfinite(row["qfim_trace_mean"]))
            self.assertGreaterEqual(row["qntk_rank"], 0)

    def test_study_is_reproducible_for_fixed_seed(self):
        kwargs = dict(
            n_qubits=2,
            n_layers=1,
            topologies=("none",),
            parameter_samples=1,
            data_points=2,
            seed=5,
        )
        first = topology_cross_metric_study(**kwargs)
        second = topology_cross_metric_study(**kwargs)
        self.assertEqual(first, second)

    def test_summary_has_one_row_per_topology(self):
        rows = topology_cross_metric_study(
            n_qubits=2,
            n_layers=1,
            topologies=("none", "line"),
            parameter_samples=2,
            data_points=2,
            seed=17,
        )
        summary = summarize_cross_metric(rows)
        self.assertEqual({row["topology"] for row in summary}, {"none", "line"})
        self.assertTrue(all(row["parameter_samples"] == 2 for row in summary))

    def test_save_cross_metric_results(self):
        rows = [{"topology": "none", "d_tp_mean": 1.0}]
        summary = [{"topology": "none", "d_tp_mean": 1.0}]
        with tempfile.TemporaryDirectory() as tmp:
            save_cross_metric_results(tmp, rows, summary, {"seed": 1})
            names = {path.name for path in Path(tmp).iterdir()}
        self.assertEqual(
            names,
            {
                "graph_topology_cross_metric_raw.csv",
                "graph_topology_cross_metric_summary.csv",
                "graph_topology_cross_metric_metadata.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
