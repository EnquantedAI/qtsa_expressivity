import unittest

import numpy as np

try:
    import pennylane as qml
except ImportError:
    qml = None

from projects.expr_train_theory.trajectory_participation.core import (
    trajectory_participation_dimension,
)
from projects.expr_train_theory.trajectory_participation.shared_qnn import (
    _prepare_inputs,
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


if __name__ == "__main__":
    unittest.main()
