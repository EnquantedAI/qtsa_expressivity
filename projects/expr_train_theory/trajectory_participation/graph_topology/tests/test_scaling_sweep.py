import tempfile
import unittest
from pathlib import Path

from projects.expr_train_theory.trajectory_participation.graph_topology.scaling_sweep import (
    iter_scaling_configs,
    save_scaling_results,
    summarize_topology_stability,
    topology_scaling_sweep,
)


class TopologyScalingSweepTests(unittest.TestCase):
    def test_iter_scaling_configs_is_cartesian_product(self):
        self.assertEqual(
            list(iter_scaling_configs((2, 3), (1, 2))),
            [(2, 1), (2, 2), (3, 1), (3, 2)],
        )

    def test_sweep_collects_each_configuration_and_topology(self):
        rows = topology_scaling_sweep(
            qubits=(2,),
            layers=(1, 2),
            topologies=("none", "line"),
            parameter_samples=1,
            data_points=1,
            seed=7,
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {(row["n_qubits"], row["n_layers"], row["topology"]) for row in rows},
            {(2, 1, "none"), (2, 1, "line"), (2, 2, "none"), (2, 2, "line")},
        )

    def test_dtp_normalization_uses_trajectory_ceiling(self):
        rows = topology_scaling_sweep(
            qubits=(2,),
            layers=(2,),
            topologies=("none",),
            parameter_samples=1,
            data_points=1,
            seed=11,
        )
        row = rows[0]
        self.assertEqual(row["trajectory_ceiling"], 3)
        self.assertAlmostEqual(
            row["d_tp_equal_normalized"], row["d_tp_equal_mean"] / 3.0
        )
        self.assertAlmostEqual(row["delta_d_tp_equal_normalized"], 0.0)

    def test_sweep_is_reproducible_for_fixed_seed(self):
        kwargs = dict(
            qubits=(2,),
            layers=(1,),
            topologies=("none", "line"),
            parameter_samples=1,
            data_points=1,
            seed=19,
        )
        self.assertEqual(topology_scaling_sweep(**kwargs), topology_scaling_sweep(**kwargs))

    def test_stability_summary_reports_sign_consistency(self):
        rows = [
            {
                "topology": "line",
                "delta_d_tp_equal_normalized": 0.10,
                "delta_d_tp_fs_normalized": 0.08,
                "delta_qfim_relative_rank_mean": 0.20,
                "delta_qntk_effective_rank": 0.30,
            },
            {
                "topology": "line",
                "delta_d_tp_equal_normalized": 0.05,
                "delta_d_tp_fs_normalized": -0.02,
                "delta_qfim_relative_rank_mean": 0.10,
                "delta_qntk_effective_rank": 0.00,
            },
            {
                "topology": "none",
                "delta_d_tp_equal_normalized": 0.0,
                "delta_d_tp_fs_normalized": 0.0,
                "delta_qfim_relative_rank_mean": 0.0,
                "delta_qntk_effective_rank": 0.0,
            },
        ]
        summary = summarize_topology_stability(rows)
        equal = next(row for row in summary if row["metric"] == "delta_d_tp_equal_normalized")
        fs = next(row for row in summary if row["metric"] == "delta_d_tp_fs_normalized")
        qntk = next(row for row in summary if row["metric"] == "delta_qntk_effective_rank")
        self.assertEqual(equal["dominant_sign"], "positive")
        self.assertAlmostEqual(equal["nonzero_sign_consistency"], 1.0)
        self.assertEqual(fs["dominant_sign"], "mixed")
        self.assertAlmostEqual(fs["nonzero_sign_consistency"], 0.5)
        self.assertAlmostEqual(qntk["zero_fraction"], 0.5)

    def test_save_writes_expected_outputs(self):
        rows = topology_scaling_sweep(
            qubits=(2,),
            layers=(1,),
            topologies=("none", "line"),
            parameter_samples=1,
            data_points=1,
            seed=5,
        )
        stability = summarize_topology_stability(rows)
        with tempfile.TemporaryDirectory() as tmp:
            save_scaling_results(tmp, rows, stability, {"test": True})
            names = {path.name for path in Path(tmp).iterdir()}
        self.assertEqual(
            names,
            {
                "graph_topology_scaling_summary.csv",
                "graph_topology_scaling_stability.csv",
                "graph_topology_scaling_metadata.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
