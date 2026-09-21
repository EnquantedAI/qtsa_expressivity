import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pennylane as qml
except ImportError:
    qml = None

from projects.expr_train_theory.trajectory_participation.native_shared_qnn_study.study import (
    NativeArchitectureConfig,
    _growth_rows,
    evaluate_trajectory,
    iter_configs,
    run_native_architecture_study,
    save_results,
    summarize_final_rows,
    summarize_growth_rows,
)


def _toy_snapshot_fn(inputs, weights, *, n_qubits, fm_style, reup_style):
    n_layers = np.asarray(weights).shape[0]
    dim = 2**n_qubits
    states = []
    for depth in range(n_layers + 1):
        angle = 0.15 * depth + 0.03 * float(np.sum(inputs))
        if depth:
            angle += 0.01 * float(np.sum(np.asarray(weights)[:depth]))
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = np.cos(angle)
        state[-1] = np.sin(angle)
        states.append(state)
    return np.asarray(states)


class NativeArchitectureConfigTests(unittest.TestCase):
    def test_qubit_count_includes_ancilla(self):
        config = NativeArchitectureConfig(2, 3, 4, "Y", "X")
        self.assertEqual(config.n_qubits, 7)

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            NativeArchitectureConfig(0, 2, 0, "Y", None)
        with self.assertRaises(ValueError):
            NativeArchitectureConfig(1, 2, -1, "Y", None)
        with self.assertRaises(ValueError):
            NativeArchitectureConfig(1, 2, 0, "bad", None)

    def test_grid_varies_depth_ancilla_encoding_and_reuploading(self):
        configs = tuple(
            iter_configs(
                layers=(1, 3),
                n_features=2,
                ancilla_counts=(0, 1),
                feature_maps=("Y", "zzfm"),
                reupload_styles=(None, "X"),
            )
        )
        self.assertEqual(len(configs), 16)
        self.assertEqual({c.n_qubits for c in configs}, {2, 3})
        self.assertEqual({c.n_layers for c in configs}, {1, 3})


class NativeArchitectureMetricTests(unittest.TestCase):
    def test_growth_starts_at_one_and_ends_at_full_trajectory(self):
        states = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.complex128,
        )
        growth = _growth_rows(states)
        self.assertEqual(len(growth), 3)
        self.assertAlmostEqual(growth[0]["d_tp_equal"], 1.0, places=12)
        self.assertAlmostEqual(growth[0]["d_tp_fs"], 1.0, places=12)
        self.assertEqual(growth[-1]["depth"], 2)
        self.assertGreaterEqual(growth[-1]["path_length_fs"], 0.0)

    def test_evaluate_trajectory_checks_native_shape(self):
        config = NativeArchitectureConfig(2, 1, 0, "Y", None)
        weights = np.zeros((2, 1, 3))
        states, final, growth = evaluate_trajectory(
            config,
            np.array([0.2]),
            weights,
            snapshot_fn=_toy_snapshot_fn,
        )
        self.assertEqual(states.shape, (3, 2))
        self.assertEqual(len(growth), 3)
        self.assertAlmostEqual(final["d_tp_equal"], growth[-1]["d_tp_equal"])


class NativeArchitectureStudyTests(unittest.TestCase):
    def test_study_reuses_matched_parameter_prefixes(self):
        seen = []

        def recording_snapshot_fn(inputs, weights, *, n_qubits, fm_style, reup_style):
            seen.append(
                (
                    n_qubits,
                    fm_style,
                    reup_style,
                    np.array(inputs, copy=True),
                    np.array(weights, copy=True),
                )
            )
            return _toy_snapshot_fn(
                inputs,
                weights,
                n_qubits=n_qubits,
                fm_style=fm_style,
                reup_style=reup_style,
            )

        rows, summary, growth, growth_summary = run_native_architecture_study(
            layers=(1, 2),
            n_features=1,
            ancilla_counts=(0, 1),
            feature_maps=("Y", "Z"),
            reupload_styles=(None,),
            samples=1,
            seed=13,
            snapshot_fn=recording_snapshot_fn,
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(summary), 8)
        self.assertGreater(len(growth), len(rows))
        self.assertGreater(len(growth_summary), len(summary))

        by_shape = {}
        for nq, fm, reup, inputs, weights in seen:
            by_shape.setdefault((nq, weights.shape[0]), []).append((inputs, weights))
        for entries in by_shape.values():
            first_input, first_weights = entries[0]
            for inputs, weights in entries[1:]:
                np.testing.assert_allclose(inputs, first_input)
                np.testing.assert_allclose(weights, first_weights)

        width_one = next(weights for nq, _, _, _, weights in seen if nq == 1 and weights.shape[0] == 1)
        width_two = next(weights for nq, _, _, _, weights in seen if nq == 2 and weights.shape[0] == 2)
        np.testing.assert_allclose(width_one, width_two[:1, :1, :])

    def test_summaries_and_saved_contract(self):
        rows, summary, growth, growth_summary = run_native_architecture_study(
            layers=(1,),
            n_features=1,
            ancilla_counts=(0,),
            feature_maps=("Y",),
            reupload_styles=(None,),
            samples=2,
            snapshot_fn=_toy_snapshot_fn,
        )
        self.assertEqual(summary, summarize_final_rows(rows))
        self.assertEqual(growth_summary, summarize_growth_rows(growth))
        self.assertEqual(summary[0]["samples"], 2)

        with tempfile.TemporaryDirectory() as tmp:
            save_results(
                tmp,
                rows,
                summary,
                growth,
                growth_summary,
                metadata={"kind": "test"},
            )
            names = {path.name for path in Path(tmp).iterdir()}
        self.assertEqual(
            names,
            {
                "native_shared_qnn_raw.csv",
                "native_shared_qnn_summary.csv",
                "native_shared_qnn_growth_raw.csv",
                "native_shared_qnn_growth_summary.csv",
                "native_shared_qnn_metadata.json",
            },
        )


@unittest.skipIf(qml is None, "PennyLane is not available")
class PennyLaneNativeArchitectureIntegrationTests(unittest.TestCase):
    def test_small_native_shared_qnn_study_runs(self):
        rows, summary, growth, growth_summary = run_native_architecture_study(
            layers=(1,),
            n_features=1,
            ancilla_counts=(0,),
            feature_maps=("Y",),
            reupload_styles=(None,),
            samples=1,
            seed=23,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(summary), 1)
        self.assertEqual(len(growth), 2)
        self.assertEqual(len(growth_summary), 2)
        self.assertTrue(np.isfinite(rows[0]["final_d_tp_equal"]))
        self.assertTrue(np.isfinite(rows[0]["final_d_tp_fs"]))


if __name__ == "__main__":
    unittest.main()
