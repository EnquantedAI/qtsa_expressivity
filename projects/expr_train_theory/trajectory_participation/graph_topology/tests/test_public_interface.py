import contextlib
import io
import json
import unittest
from pathlib import Path

from projects.expr_train_theory.trajectory_participation import graph_topology
from projects.expr_train_theory.trajectory_participation.graph_topology import run_reproducible_study


class GraphTopologyPublicInterfaceTests(unittest.TestCase):
    def test_package_exports_only_reproducible_study_surface(self):
        self.assertEqual(
            graph_topology.__all__,
            [
                "DEFAULT_REPRODUCIBLE_PRESET",
                "REPRODUCIBLE_STUDY_NAME",
                "TopologyStudyPreset",
                "preset_manifest",
                "run_reproducible_topology_study",
                "save_reproducible_topology_study",
            ],
        )

    def test_package_default_preset_matches_public_name(self):
        self.assertEqual(
            graph_topology.DEFAULT_REPRODUCIBLE_PRESET.name,
            graph_topology.REPRODUCIBLE_STUDY_NAME,
        )
        self.assertEqual(graph_topology.REPRODUCIBLE_STUDY_NAME, "topology_robust_v1")

    def test_default_output_directory_uses_frozen_preset_name(self):
        output = run_reproducible_study.default_output_dir()
        self.assertEqual(output.name, "topology_robust_v1")
        self.assertEqual(output.parent.name, "results")

    def test_cli_accepts_custom_output_directory(self):
        args = run_reproducible_study.build_parser().parse_args(
            ["--output-dir", "/tmp/topology-release"]
        )
        self.assertEqual(args.output_dir, Path("/tmp/topology-release"))
        self.assertFalse(args.show_preset)

    def test_show_preset_prints_manifest_without_running_study(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            return_code = run_reproducible_study.main(["--show-preset"])
        manifest = json.loads(stream.getvalue())
        self.assertEqual(return_code, 0)
        self.assertEqual(manifest["name"], "topology_robust_v1")
        self.assertEqual(manifest["repeat_seeds"][0], manifest["base_seed"])
        self.assertIn("run_reproducible_study", manifest["runner"])

    def test_cli_help_is_available_without_running_study(self):
        stream = io.StringIO()
        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stdout(stream):
            run_reproducible_study.main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--show-preset", stream.getvalue())
        self.assertIn("--output-dir", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
