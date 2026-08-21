import tempfile
import unittest
from pathlib import Path

from projects.expr_train_theory.trajectory_participation.graph_topology.uncertainty_sweep import (
    bootstrap_matched_delta_uncertainty,
    repeat_seeds,
    save_uncertainty_results,
    summarize_ci_stability,
    topology_scaling_multiseed,
)


class TopologyUncertaintySweepTests(unittest.TestCase):
    def test_repeat_seeds_are_deterministic_and_separated(self):
        self.assertEqual(repeat_seeds(base_seed=7, repeats=3, stride=100), (7, 107, 207))
        with self.assertRaises(ValueError):
            repeat_seeds(repeats=0)

    def test_multiseed_sweep_annotates_repeat_identity(self):
        rows = topology_scaling_multiseed(
            qubits=(2,),
            layers=(1,),
            topologies=("none", "line"),
            parameter_samples=1,
            data_points=1,
            base_seed=5,
            repeats=2,
            seed_stride=100,
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["repeat_index"] for row in rows}, {0, 1})
        self.assertEqual({row["repeat_seed"] for row in rows}, {5, 105})

    def test_bootstrap_constant_delta_has_zero_width_interval(self):
        rows = []
        for repeat_index in range(4):
            rows.append(
                {
                    "n_qubits": 2,
                    "n_layers": 1,
                    "topology": "line",
                    "repeat_index": repeat_index,
                    "delta_d_tp_equal_normalized": 0.25,
                }
            )
        report = bootstrap_matched_delta_uncertainty(
            rows,
            metrics=("delta_d_tp_equal_normalized",),
            bootstrap_samples=100,
            seed=3,
        )
        self.assertEqual(len(report), 1)
        self.assertAlmostEqual(report[0]["mean_delta"], 0.25)
        self.assertAlmostEqual(report[0]["ci_lower"], 0.25)
        self.assertAlmostEqual(report[0]["ci_upper"], 0.25)
        self.assertEqual(report[0]["ci_relation"], "positive")

    def test_baseline_uncertainty_is_exactly_zero(self):
        rows = topology_scaling_multiseed(
            qubits=(2,),
            layers=(1,),
            topologies=("none", "line"),
            parameter_samples=1,
            data_points=1,
            base_seed=13,
            repeats=2,
            seed_stride=100,
        )
        report = bootstrap_matched_delta_uncertainty(
            rows,
            metrics=("delta_d_tp_equal_normalized",),
            bootstrap_samples=50,
            seed=9,
        )
        baseline = next(row for row in report if row["topology"] == "none")
        self.assertAlmostEqual(baseline["mean_delta"], 0.0)
        self.assertAlmostEqual(baseline["ci_lower"], 0.0)
        self.assertAlmostEqual(baseline["ci_upper"], 0.0)
        self.assertEqual(baseline["ci_relation"], "overlaps_zero")

    def test_ci_stability_keeps_unresolved_configurations_separate(self):
        rows = [
            {"topology": "line", "metric": "m", "ci_relation": "positive"},
            {"topology": "line", "metric": "m", "ci_relation": "overlaps_zero"},
            {"topology": "line", "metric": "m", "ci_relation": "positive"},
            {"topology": "star", "metric": "m", "ci_relation": "positive"},
            {"topology": "star", "metric": "m", "ci_relation": "negative"},
        ]
        summary = summarize_ci_stability(rows)
        line = next(row for row in summary if row["topology"] == "line")
        star = next(row for row in summary if row["topology"] == "star")
        self.assertAlmostEqual(line["resolved_fraction"], 2.0 / 3.0)
        self.assertEqual(line["dominant_resolved_sign"], "positive")
        self.assertAlmostEqual(line["resolved_sign_consistency"], 1.0)
        self.assertEqual(star["dominant_resolved_sign"], "mixed")
        self.assertAlmostEqual(star["resolved_sign_consistency"], 0.5)

    def test_save_writes_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_uncertainty_results(
                tmp,
                [{"raw": 1}],
                [{"uncertainty": 1}],
                [{"stability": 1}],
                {"test": True},
            )
            names = {path.name for path in Path(tmp).iterdir()}
        self.assertEqual(
            names,
            {
                "graph_topology_multiseed_raw.csv",
                "graph_topology_multiseed_uncertainty.csv",
                "graph_topology_multiseed_ci_stability.csv",
                "graph_topology_multiseed_metadata.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
