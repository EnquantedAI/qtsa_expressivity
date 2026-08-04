import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.krylov.circuit_orbits.circuits import (
    identity,
    layered_ansatz,
    pauli_x,
)
from projects.expr_train_theory.krylov.circuit_orbits.study import (
    analyse_layer_orbit,
    cumulative_states,
    orthonormal_span,
    run_parameter_study,
)


class CircuitOrbitTests(unittest.TestCase):
    def test_identity_layers_stay_one_dimensional(self) -> None:
        state = np.array([1.0, 0.0], dtype=np.complex128)
        result = analyse_layer_orbit([identity(1), identity(1)], state)
        self.assertEqual(result.layer_orbit_dimension, 1)
        self.assertEqual(result.repeated_cycle_dimension, 1)
        self.assertAlmostEqual(result.projector_distance, 0.0, places=12)

    def test_single_x_layer_reaches_two_directions(self) -> None:
        state = np.array([1.0, 0.0], dtype=np.complex128)
        result = analyse_layer_orbit([pauli_x()], state)
        self.assertEqual(result.layer_orbit_dimension, 2)
        self.assertEqual(result.repeated_cycle_dimension, 2)
        self.assertAlmostEqual(result.projector_distance, 0.0, places=12)

    def test_two_x_layers_have_two_dimensional_layer_span(self) -> None:
        state = np.array([1.0, 0.0], dtype=np.complex128)
        states = cumulative_states([pauli_x(), pauli_x()], state)
        basis = orthonormal_span(states)
        self.assertEqual(basis.shape[1], 2)

    def test_layered_ansatz_returns_unitaries(self) -> None:
        parameters = np.zeros((2, 2, 3), dtype=float)
        layers = layered_ansatz(parameters, n_qubits=2)
        self.assertEqual(len(layers), 2)
        for layer in layers:
            np.testing.assert_allclose(layer.conj().T @ layer, np.eye(4), atol=1e-12)

    def test_non_unitary_layer_is_rejected(self) -> None:
        state = np.array([1.0, 0.0], dtype=np.complex128)
        with self.assertRaises(ValueError):
            cumulative_states([np.diag([1.0, 2.0])], state)

    def test_parameter_study_writes_results(self) -> None:
        state = np.zeros(4, dtype=np.complex128)
        state[0] = 1.0
        parameter_sets = [np.zeros((1, 2, 3)), np.zeros((2, 2, 3))]
        with tempfile.TemporaryDirectory() as directory:
            results = run_parameter_study(
                parameter_sets,
                n_qubits=2,
                initial_state=state,
                layer_builder=layered_ansatz,
                output_directory=directory,
            )
            self.assertEqual(len(results), 2)
            self.assertTrue((Path(directory) / "circuit_orbit_results.csv").exists())
            self.assertTrue((Path(directory) / "circuit_orbit_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
