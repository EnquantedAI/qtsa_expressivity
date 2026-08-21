import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.trajectory_participation.graph_topology.fs_weighting import (
    matched_topology_weighting_deltas,
    save_topology_weighting_results,
    summarize_topology_weighting,
    topology_fs_weighting_study,
)


class FsTopologyWeightingTests(unittest.TestCase):
    def test_study_contains_equal_and_fs_metrics(self):
        rows = topology_fs_weighting_study(
            n_qubits=2,
            n_layers=1,
            topologies=("none", "line"),
            parameter_samples=1,
            data_points=2,
            seed=7,
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertGreaterEqual(row["d_tp_equal_mean"], 1.0)
            self.assertGreaterEqual(row["d_tp_fs_mean"], 1.0)
            self.assertGreaterEqual(row["path_length_fs_mean"], 0.0)

    def test_study_is_deterministic_for_fixed_seed(self):
        kwargs = dict(
            n_qubits=2,
            n_layers=1,
            topologies=("none", "line"),
            parameter_samples=1,
            data_points=2,
            seed=11,
        )
        first = topology_fs_weighting_study(**kwargs)
        second = topology_fs_weighting_study(**kwargs)
        for a, b in zip(first, second):
            self.assertAlmostEqual(a["d_tp_equal_mean"], b["d_tp_equal_mean"])
            self.assertAlmostEqual(a["d_tp_fs_mean"], b["d_tp_fs_mean"])

    def test_matched_baseline_deltas_are_zero(self):
        rows = topology_fs_weighting_study(
            n_qubits=2,
            n_layers=1,
            topologies=("none", "line"),
            parameter_samples=1,
            data_points=1,
            seed=3,
        )
        deltas = matched_topology_weighting_deltas(rows)
        baseline = next(row for row in deltas if row["topology"] == "none")
        self.assertAlmostEqual(baseline["delta_d_tp_equal"], 0.0)
        self.assertAlmostEqual(baseline["delta_d_tp_fs"], 0.0)
        self.assertAlmostEqual(baseline["delta_fs_minus_equal"], 0.0)

    def test_fs_effect_delta_is_difference_of_topology_deltas(self):
        rows = [
            {
                "topology": "none", "parameter_sample": 0, "n_edges": 0,
                "density": 0.0, "algebraic_connectivity": 0.0,
                "d_tp_equal_mean": 1.2, "d_tp_fs_mean": 1.1,
                "path_length_fs_mean": 0.5,
            },
            {
                "topology": "line", "parameter_sample": 0, "n_edges": 1,
                "density": 1.0, "algebraic_connectivity": 2.0,
                "d_tp_equal_mean": 1.7, "d_tp_fs_mean": 1.4,
                "path_length_fs_mean": 0.9,
            },
        ]
        line = matched_topology_weighting_deltas(rows)[1]
        self.assertAlmostEqual(line["delta_d_tp_equal"], 0.5)
        self.assertAlmostEqual(line["delta_d_tp_fs"], 0.3)
        self.assertAlmostEqual(line["delta_fs_minus_equal"], -0.2)

    def test_summary_preserves_matched_effect(self):
        rows = topology_fs_weighting_study(
            n_qubits=2,
            n_layers=1,
            topologies=("none", "line"),
            parameter_samples=2,
            data_points=1,
            seed=13,
        )
        deltas = matched_topology_weighting_deltas(rows)
        summary = summarize_topology_weighting(rows, deltas)
        line = next(row for row in summary if row["topology"] == "line")
        self.assertTrue(np.isfinite(line["delta_fs_minus_equal_mean"]))
        self.assertAlmostEqual(
            line["delta_fs_minus_equal_mean"],
            line["delta_d_tp_fs_mean"] - line["delta_d_tp_equal_mean"],
        )

    def test_save_writes_expected_outputs(self):
        rows = topology_fs_weighting_study(
            n_qubits=2,
            n_layers=1,
            topologies=("none",),
            parameter_samples=1,
            data_points=1,
            seed=5,
        )
        deltas = matched_topology_weighting_deltas(rows)
        summary = summarize_topology_weighting(rows, deltas)
        with tempfile.TemporaryDirectory() as tmp:
            save_topology_weighting_results(tmp, rows, deltas, summary, {"test": True})
            names = {path.name for path in Path(tmp).iterdir()}
        self.assertEqual(
            names,
            {
                "graph_topology_fs_weighting_raw.csv",
                "graph_topology_fs_weighting_deltas.csv",
                "graph_topology_fs_weighting_summary.csv",
                "graph_topology_fs_weighting_metadata.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
