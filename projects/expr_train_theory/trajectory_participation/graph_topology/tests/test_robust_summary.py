import tempfile
import unittest
from pathlib import Path

from projects.expr_train_theory.trajectory_participation.graph_topology.robust_summary import (
    build_robust_topology_summary,
    classify_uncertainty_rows,
    save_robust_summary,
    summarize_robust_effects,
)


class RobustTopologySummaryTests(unittest.TestCase):
    def test_configuration_classification_uses_full_interval_relation(self):
        rows = [
            {
                "n_qubits": 3,
                "n_layers": 2,
                "topology": "line",
                "metric": "m",
                "mean_delta": 0.2,
                "ci_relation": "positive",
            },
            {
                "n_qubits": 3,
                "n_layers": 3,
                "topology": "line",
                "metric": "m",
                "mean_delta": -0.2,
                "ci_relation": "negative",
            },
            {
                "n_qubits": 4,
                "n_layers": 2,
                "topology": "line",
                "metric": "m",
                "mean_delta": 0.01,
                "ci_relation": "overlaps_zero",
            },
        ]
        classified = classify_uncertainty_rows(rows)
        self.assertEqual(
            [row["effect_class"] for row in classified],
            ["stable_positive", "stable_negative", "unresolved"],
        )

    def test_baseline_is_not_reported_as_a_topology_effect(self):
        rows = [
            {
                "n_qubits": 2,
                "n_layers": 1,
                "topology": "none",
                "metric": "m",
                "mean_delta": 0.0,
                "ci_relation": "overlaps_zero",
            }
        ]
        self.assertEqual(classify_uncertainty_rows(rows), [])

    def test_all_positive_configurations_give_stable_positive_summary(self):
        classified = [
            {
                "topology": "line",
                "metric": "m",
                "effect_class": "stable_positive",
                "mean_delta": 0.1,
            },
            {
                "topology": "line",
                "metric": "m",
                "effect_class": "stable_positive",
                "mean_delta": 0.2,
            },
        ]
        summary = summarize_robust_effects(classified)
        self.assertEqual(summary[0]["robust_classification"], "stable_positive")
        self.assertEqual(summary[0]["resolved_direction"], "positive")
        self.assertAlmostEqual(summary[0]["resolved_fraction"], 1.0)

    def test_unresolved_configuration_keeps_aggregate_label_conservative(self):
        classified = [
            {
                "topology": "line",
                "metric": "m",
                "effect_class": "stable_positive",
                "mean_delta": 0.1,
            },
            {
                "topology": "line",
                "metric": "m",
                "effect_class": "unresolved",
                "mean_delta": 0.02,
            },
        ]
        summary = summarize_robust_effects(classified)
        self.assertEqual(summary[0]["robust_classification"], "unresolved")
        self.assertEqual(summary[0]["resolved_direction"], "positive")
        self.assertAlmostEqual(summary[0]["resolved_fraction"], 0.5)

    def test_opposite_resolved_signs_are_reported_as_mixed_and_unresolved(self):
        classified = [
            {
                "topology": "star",
                "metric": "m",
                "effect_class": "stable_positive",
                "mean_delta": 0.1,
            },
            {
                "topology": "star",
                "metric": "m",
                "effect_class": "stable_negative",
                "mean_delta": -0.1,
            },
        ]
        summary = summarize_robust_effects(classified)
        self.assertEqual(summary[0]["resolved_direction"], "mixed")
        self.assertEqual(summary[0]["robust_classification"], "unresolved")
        self.assertAlmostEqual(summary[0]["resolved_sign_consistency"], 0.5)

    def test_build_and_save_write_expected_tables(self):
        uncertainty = [
            {
                "n_qubits": 3,
                "n_layers": 2,
                "topology": "line",
                "metric": "m",
                "mean_delta": 0.2,
                "ci_relation": "positive",
            }
        ]
        configurations, summary = build_robust_topology_summary(uncertainty)
        with tempfile.TemporaryDirectory() as tmp:
            save_robust_summary(tmp, configurations, summary, {"test": True})
            names = {path.name for path in Path(tmp).iterdir()}
        self.assertEqual(
            names,
            {
                "graph_topology_robust_configurations.csv",
                "graph_topology_robust_summary.csv",
                "graph_topology_robust_metadata.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
