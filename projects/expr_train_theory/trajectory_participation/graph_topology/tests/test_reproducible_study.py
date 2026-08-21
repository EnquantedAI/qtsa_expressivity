import json
import tempfile
import unittest
from pathlib import Path

from projects.expr_train_theory.trajectory_participation.graph_topology.reproducible_study import (
    DEFAULT_REPRODUCIBLE_PRESET,
    REPRODUCIBLE_STUDY_NAME,
    TopologyStudyPreset,
    preset_manifest,
    run_reproducible_topology_study,
    save_reproducible_topology_study,
)


class ReproducibleTopologyStudyTests(unittest.TestCase):
    def small_preset(self):
        return TopologyStudyPreset(
            name="test",
            qubits=(2,),
            layers=(1,),
            topologies=("none", "line"),
            parameter_samples=1,
            data_points=1,
            base_seed=11,
            repeats=2,
            seed_stride=100,
            bootstrap_samples=30,
            confidence=0.9,
        )

    def test_default_preset_is_the_named_official_configuration(self):
        preset = DEFAULT_REPRODUCIBLE_PRESET
        self.assertEqual(preset.name, REPRODUCIBLE_STUDY_NAME)
        self.assertEqual(preset.qubits, (3, 4))
        self.assertEqual(preset.layers, (1, 2, 3))
        self.assertEqual(preset.repeats, 8)
        self.assertEqual(preset.base_seed, 2026)

    def test_preset_validation_rejects_missing_baseline(self):
        preset = TopologyStudyPreset(topologies=("line",), baseline_topology="none")
        with self.assertRaises(ValueError):
            preset.validate()

    def test_custom_baseline_is_used_by_the_complete_pipeline(self):
        preset = TopologyStudyPreset(
            name="custom-baseline",
            qubits=(2,),
            layers=(1,),
            topologies=("none", "line"),
            baseline_topology="line",
            parameter_samples=1,
            data_points=1,
            base_seed=19,
            repeats=2,
            seed_stride=100,
            bootstrap_samples=20,
            confidence=0.9,
        )
        result = run_reproducible_topology_study(preset)

        line_rows = [
            row for row in result["repeat_rows"] if row["topology"] == "line"
        ]
        self.assertTrue(line_rows)
        self.assertTrue(
            all(row["baseline_topology"] == "line" for row in result["repeat_rows"])
        )
        for row in line_rows:
            self.assertAlmostEqual(row["delta_d_tp_equal_normalized"], 0.0)
            self.assertAlmostEqual(row["delta_d_tp_fs_normalized"], 0.0)
            self.assertAlmostEqual(row["delta_qfim_relative_rank_mean"], 0.0)
            self.assertAlmostEqual(row["delta_qntk_effective_rank"], 0.0)

    def test_manifest_records_exact_repeat_and_bootstrap_seeds(self):
        manifest = preset_manifest(self.small_preset())
        self.assertEqual(manifest["repeat_seeds"], [11, 111])
        self.assertEqual(manifest["bootstrap_seed"], 788)
        self.assertIn("run_reproducible_study", manifest["runner"])

    def test_small_reproducible_study_contains_all_pipeline_layers(self):
        result = run_reproducible_topology_study(self.small_preset())
        self.assertTrue(result["repeat_rows"])
        self.assertTrue(result["uncertainty_rows"])
        self.assertTrue(result["ci_stability_rows"])
        self.assertTrue(result["robust_configuration_rows"])
        self.assertTrue(result["robust_summary_rows"])

    def test_same_preset_is_deterministic(self):
        preset = self.small_preset()
        first = run_reproducible_topology_study(preset)
        second = run_reproducible_topology_study(preset)
        self.assertEqual(first["manifest"], second["manifest"])
        self.assertEqual(first["repeat_rows"], second["repeat_rows"])
        self.assertEqual(first["uncertainty_rows"], second["uncertainty_rows"])
        self.assertEqual(
            [row["robust_classification"] for row in first["robust_summary_rows"]],
            [row["robust_classification"] for row in second["robust_summary_rows"]],
        )
        self.assertEqual(
            [row["mean_delta_across_configurations"] for row in first["robust_summary_rows"]],
            [row["mean_delta_across_configurations"] for row in second["robust_summary_rows"]],
        )

    def test_save_writes_complete_release_table_set_and_manifest(self):
        result = run_reproducible_topology_study(self.small_preset())
        with tempfile.TemporaryDirectory() as tmp:
            names = set(save_reproducible_topology_study(tmp, result))
            manifest_path = Path(tmp) / "graph_topology_reproducible_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            names,
            {
                "graph_topology_multiseed_raw.csv",
                "graph_topology_multiseed_uncertainty.csv",
                "graph_topology_multiseed_ci_stability.csv",
                "graph_topology_multiseed_metadata.json",
                "graph_topology_robust_configurations.csv",
                "graph_topology_robust_summary.csv",
                "graph_topology_robust_metadata.json",
                "graph_topology_reproducible_manifest.json",
            },
        )
        self.assertEqual(manifest["name"], "test")
        self.assertEqual(manifest["repeat_seeds"], [11, 111])


if __name__ == "__main__":
    unittest.main()
