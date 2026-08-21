import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.trajectory_participation.graph_topology.comprehensive import (
    matched_comprehensive_deltas,
    save_comprehensive_results,
    summarize_comprehensive,
    topology_comprehensive_study,
)


class ComprehensiveTopologyTests(unittest.TestCase):
    def test_study_collects_all_main_metric_families(self):
        rows = topology_comprehensive_study(
            n_qubits=2,
            n_layers=1,
            topologies=("none", "line"),
            parameter_samples=1,
            data_points=2,
            seed=7,
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            for key in (
                "d_tp_equal_mean",
                "d_tp_fs_mean",
                "path_length_fs_mean",
                "qfim_rank_mean",
                "qfim_trace_mean",
                "qntk_rank",
                "qntk_effective_rank",
                "density",
                "algebraic_connectivity",
            ):
                self.assertIn(key, row)
            self.assertTrue(np.isfinite(row["d_tp_equal_mean"]))
            self.assertTrue(np.isfinite(row["d_tp_fs_mean"]))
            self.assertTrue(np.isfinite(row["qfim_trace_mean"]))

    def test_study_is_reproducible_for_fixed_seed(self):
        kwargs = dict(
            n_qubits=2,
            n_layers=1,
            topologies=("none", "line"),
            parameter_samples=1,
            data_points=2,
            seed=19,
        )
        self.assertEqual(
            topology_comprehensive_study(**kwargs),
            topology_comprehensive_study(**kwargs),
        )

    def test_baseline_matched_deltas_are_zero(self):
        rows = topology_comprehensive_study(
            n_qubits=2,
            n_layers=1,
            topologies=("none", "line"),
            parameter_samples=1,
            data_points=1,
            seed=3,
        )
        deltas = matched_comprehensive_deltas(rows)
        baseline = next(row for row in deltas if row["topology"] == "none")
        self.assertAlmostEqual(baseline["delta_d_tp_equal_mean"], 0.0)
        self.assertAlmostEqual(baseline["delta_d_tp_fs_mean"], 0.0)
        self.assertAlmostEqual(baseline["delta_qfim_trace_mean"], 0.0)
        self.assertAlmostEqual(baseline["delta_qntk_effective_rank"], 0.0)
        self.assertAlmostEqual(baseline["delta_fs_minus_equal"], 0.0)

    def test_matched_delta_uses_same_parameter_sample(self):
        def row(topology, sample, value):
            return {
                "topology": topology,
                "parameter_sample": sample,
                "n_edges": 0 if topology == "none" else 1,
                "density": 0.0 if topology == "none" else 1.0,
                "mean_degree": 0.0 if topology == "none" else 1.0,
                "max_degree": 0 if topology == "none" else 1,
                "connected_components": 2 if topology == "none" else 1,
                "diameter": float("inf") if topology == "none" else 1.0,
                "mean_shortest_path": float("inf") if topology == "none" else 1.0,
                "algebraic_connectivity": 0.0 if topology == "none" else 2.0,
                "d_tp_equal_mean": value,
                "d_tp_fs_mean": value + 0.1,
                "path_length_fs_mean": value + 0.2,
                "trajectory_rank_mean": value + 1.0,
                "qfim_rank_mean": value + 2.0,
                "qfim_relative_rank_mean": value + 0.3,
                "qfim_trace_mean": value + 3.0,
                "qntk_rank": value + 4.0,
                "qntk_effective_rank": value + 5.0,
                "qntk_trace": value + 6.0,
            }

        rows = [
            row("none", 0, 1.0),
            row("none", 1, 10.0),
            row("line", 0, 1.4),
            row("line", 1, 10.7),
        ]
        line = [r for r in matched_comprehensive_deltas(rows) if r["topology"] == "line"]
        self.assertAlmostEqual(line[0]["delta_d_tp_equal_mean"], 0.4)
        self.assertAlmostEqual(line[1]["delta_d_tp_equal_mean"], 0.7)

    def test_summary_keeps_weighting_effect_identity(self):
        rows = topology_comprehensive_study(
            n_qubits=2,
            n_layers=1,
            topologies=("none", "line"),
            parameter_samples=2,
            data_points=1,
            seed=13,
        )
        deltas = matched_comprehensive_deltas(rows)
        summary = summarize_comprehensive(rows, deltas)
        line = next(row for row in summary if row["topology"] == "line")
        self.assertAlmostEqual(
            line["delta_fs_minus_equal"],
            line["delta_d_tp_fs_mean"] - line["delta_d_tp_equal_mean"],
        )
        self.assertEqual(line["parameter_samples"], 2)

    def test_save_writes_expected_outputs(self):
        rows = topology_comprehensive_study(
            n_qubits=2,
            n_layers=1,
            topologies=("none",),
            parameter_samples=1,
            data_points=1,
            seed=5,
        )
        deltas = matched_comprehensive_deltas(rows)
        summary = summarize_comprehensive(rows, deltas)
        with tempfile.TemporaryDirectory() as tmp:
            save_comprehensive_results(tmp, rows, deltas, summary, {"test": True})
            names = {path.name for path in Path(tmp).iterdir()}
        self.assertEqual(
            names,
            {
                "graph_topology_comprehensive_raw.csv",
                "graph_topology_comprehensive_deltas.csv",
                "graph_topology_comprehensive_summary.csv",
                "graph_topology_comprehensive_metadata.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
