import sys
import types
import unittest
from unittest import mock

import numpy as np

try:
    import pennylane as qml
except ImportError:
    qml = None

from projects.expr_train_theory.trajectory_participation.core import (
    trajectory_participation_dimension,
)
import projects.expr_train_theory.trajectory_participation.shared_qnn as shared_qnn_module

from projects.expr_train_theory.trajectory_participation.shared_qnn import (
    _extract_depth_snapshots,
    _prepare_inputs,
    native_shared_qnn_snapshots,
    reference_shared_qnn_snapshots,
    shared_qnn_snapshots,
    z_expectation_from_state,
)


class SharedQNNInputTests(unittest.TestCase):
    def test_long_input_keeps_most_recent_values(self):
        values = _prepare_inputs([1.0, 2.0, 3.0, 4.0], 2)
        np.testing.assert_allclose(values, [3.0, 4.0])

    def test_short_input_is_not_padded(self):
        values = _prepare_inputs([1.0, 2.0], 4)
        np.testing.assert_allclose(values, [1.0, 2.0])

    def test_z_expectation(self):
        self.assertAlmostEqual(z_expectation_from_state([1.0, 0.0], 0, 1), 1.0)
        self.assertAlmostEqual(z_expectation_from_state([0.0, 1.0], 0, 1), -1.0)


class NativeSnapshotExtractionTests(unittest.TestCase):
    def test_depth_snapshots_are_selected_in_depth_order(self):
        snapshots = {
            "execution_results": [0.0],
            "depth_1": np.array([0.0, 1.0, 0.0, 0.0]),
            "depth_0": np.array([1.0, 0.0, 0.0, 0.0]),
        }
        states = _extract_depth_snapshots(snapshots, n_layers=1, n_qubits=2)
        self.assertEqual(states.shape, (2, 4))
        np.testing.assert_allclose(states[0], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(states[1], [0.0, 1.0, 0.0, 0.0])

    def test_missing_depth_snapshot_is_reported(self):
        with self.assertRaisesRegex(KeyError, "depth_1"):
            _extract_depth_snapshots(
                {"depth_0": np.array([1.0, 0.0])},
                n_layers=1,
                n_qubits=1,
            )

    def test_wrong_snapshot_dimension_is_reported(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            _extract_depth_snapshots(
                {"depth_0": np.array([1.0, 0.0, 0.0])},
                n_layers=0,
                n_qubits=1,
            )

    def test_native_adapter_delegates_to_shared_gqnn_snapshot_hooks(self):
        calls = {}

        def fake_gqnn(n_layers, n_qubits, dev, **kwargs):
            calls["gqnn"] = {
                "n_layers": n_layers,
                "n_qubits": n_qubits,
                "dev": dev,
                **kwargs,
            }
            return object()

        def fake_snapshots(circuit):
            calls["circuit"] = circuit

            def execute(inputs, weights):
                calls["inputs"] = np.asarray(inputs)
                calls["weights"] = np.asarray(weights)
                return {
                    "depth_0": np.array([1.0, 0.0]),
                    "depth_1": np.array([0.0, 1.0]),
                    "execution_results": [0.0],
                }

            return execute

        fake_qml = types.SimpleNamespace(
            device=lambda name, wires: (name, wires),
            snapshots=fake_snapshots,
        )
        fake_models = types.ModuleType("src.models")
        fake_models.gqnn = fake_gqnn
        weights = np.zeros((1, 1, 3))

        with mock.patch.object(shared_qnn_module, "qml", fake_qml), mock.patch.dict(
            sys.modules, {"src.models": fake_models}
        ):
            states = native_shared_qnn_snapshots(
                [0.1, 0.2],
                weights,
                n_qubits=1,
                fm_style="Y",
                reup_style="X",
            )

        self.assertTrue(calls["gqnn"]["add_snaps"])
        self.assertEqual(calls["gqnn"]["fm_style"], "Y")
        self.assertEqual(calls["gqnn"]["reup_style"], "X")
        self.assertEqual(calls["inputs"].shape, (2,))
        np.testing.assert_allclose(states, [[1.0, 0.0], [0.0, 1.0]])


@unittest.skipIf(qml is None, "PennyLane is not available")
class SharedQNNTrajectoryTests(unittest.TestCase):
    def test_snapshot_count_and_norm(self):
        weights = np.zeros((3, 2, 3))
        states = shared_qnn_snapshots([0.2, 0.4], weights, n_qubits=2, fm_style="Y")
        self.assertEqual(states.shape, (4, 4))
        np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-10)

    def test_extra_qubit_is_supported(self):
        weights = np.zeros((2, 3, 3))
        states = shared_qnn_snapshots([0.2, 0.4], weights, n_qubits=3, fm_style="Y")
        self.assertEqual(states.shape, (3, 8))

    def test_reuploading_changes_generic_trajectory(self):
        rng = np.random.default_rng(9)
        weights = rng.normal(size=(3, 2, 3))
        plain = shared_qnn_snapshots([0.3, 0.8], weights, n_qubits=2, fm_style="Y")
        reup = shared_qnn_snapshots(
            [0.3, 0.8], weights, n_qubits=2, fm_style="Y", reup_style="Y"
        )
        self.assertFalse(np.allclose(plain, reup))

    def test_dimension_stays_in_expected_range(self):
        rng = np.random.default_rng(17)
        weights = rng.normal(size=(4, 2, 3))
        states = shared_qnn_snapshots([0.1, 0.6], weights, n_qubits=2)
        result = trajectory_participation_dimension(states)
        self.assertGreaterEqual(result.dimension, 1.0 - 1e-10)
        self.assertLessEqual(result.dimension, min(states.shape) + 1e-10)

    def test_final_state_matches_shared_model_measurement(self):
        from src.models import gqnn

        rng = np.random.default_rng(31)
        n_qubits = 2
        n_layers = 2
        inputs = np.array([0.2, 0.9])
        weights = rng.normal(size=(n_layers, n_qubits, 3))

        states = shared_qnn_snapshots(
            inputs,
            weights,
            n_qubits=n_qubits,
            fm_style="zzfm",
            reup_style="Y",
        )

        dev = qml.device("default.qubit", wires=n_qubits)
        model = gqnn(
            n_layers,
            n_qubits,
            dev,
            fm_style="zzfm",
            reup_style="Y",
            meas=[0, 1],
        )
        expected = np.asarray(model(inputs, weights), dtype=float)
        observed = np.array(
            [z_expectation_from_state(states[-1], wire, n_qubits) for wire in range(n_qubits)]
        )
        np.testing.assert_allclose(observed, expected, atol=1e-8)

    def test_final_state_matches_batched_model_for_one_sample(self):
        from src.models_batch import gqnn

        rng = np.random.default_rng(41)
        n_qubits = 3
        n_layers = 2
        inputs = np.array([0.15, 0.45])
        weights = rng.normal(size=(n_layers, n_qubits, 3))

        states = shared_qnn_snapshots(
            inputs,
            weights,
            n_qubits=n_qubits,
            fm_style="Y",
            reup_style="X",
        )

        dev = qml.device("default.qubit", wires=n_qubits)
        model = gqnn(
            n_layers,
            n_qubits,
            dev,
            fm_style="Y",
            reup_style="X",
            meas=[0, 2],
        )
        batch_inputs = inputs.reshape(1, -1)
        expected = np.asarray(model(batch_inputs, weights), dtype=float).reshape(2, -1)[:, 0]
        observed = np.array(
            [z_expectation_from_state(states[-1], wire, n_qubits) for wire in (0, 2)]
        )
        np.testing.assert_allclose(observed, expected, atol=1e-8)


    def test_native_snapshots_come_from_shared_model(self):
        rng = np.random.default_rng(53)
        n_qubits = 3
        n_layers = 2
        inputs = np.array([0.2, 0.5])
        weights = rng.normal(size=(n_layers, n_qubits, 3))

        states = native_shared_qnn_snapshots(
            inputs,
            weights,
            n_qubits=n_qubits,
            fm_style="Y",
            reup_style="X",
        )
        self.assertEqual(states.shape, (n_layers + 1, 2**n_qubits))
        np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-10)

    def test_native_and_reference_paths_match_for_shared_layout(self):
        rng = np.random.default_rng(59)
        n_qubits = 3
        n_layers = 3
        inputs = np.array([0.15, 0.45])
        weights = rng.normal(size=(n_layers, n_qubits, 3))

        native = native_shared_qnn_snapshots(
            inputs,
            weights,
            n_qubits=n_qubits,
            fm_style="Y",
            reup_style="Z",
        )
        reference = reference_shared_qnn_snapshots(
            inputs,
            weights,
            n_qubits=n_qubits,
            fm_style="Y",
            reup_style="Z",
        )
        np.testing.assert_allclose(native, reference, atol=1e-8)
        self.assertAlmostEqual(
            trajectory_participation_dimension(native).dimension,
            trajectory_participation_dimension(reference).dimension,
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
