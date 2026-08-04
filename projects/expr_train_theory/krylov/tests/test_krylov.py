import unittest

import numpy as np

from projects.expr_train_theory.krylov.core import arnoldi, lanczos
from projects.expr_train_theory.krylov.dynamics import exact_state, projected_state
from projects.expr_train_theory.krylov.metrics import (
    krylov_entropy,
    participation_ratio,
    spread_complexity,
    state_probabilities,
)
from projects.expr_train_theory.krylov.models import SIGMA_X, basis_state, path_hamiltonian


class LanczosTests(unittest.TestCase):
    def test_eigenvector_gives_one_dimensional_space(self):
        hamiltonian = np.diag([2.0, -1.0])
        result = lanczos(hamiltonian, basis_state(0, 2))
        self.assertEqual(result.dimension, 1)
        np.testing.assert_allclose(result.alpha, [2.0], atol=1e-12)
        self.assertEqual(result.beta.size, 0)

    def test_pauli_x_has_known_tridiagonal_form(self):
        result = lanczos(SIGMA_X, basis_state(0, 2))
        self.assertEqual(result.dimension, 2)
        np.testing.assert_allclose(result.tridiagonal, SIGMA_X, atol=1e-12)
        np.testing.assert_allclose(result.alpha, [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(result.beta, [1.0], atol=1e-12)

    def test_path_hamiltonian_spans_full_space(self):
        hamiltonian = path_hamiltonian(4)
        result = lanczos(hamiltonian, basis_state(0, 4))
        self.assertEqual(result.dimension, 4)
        np.testing.assert_allclose(result.alpha, np.zeros(4), atol=1e-12)
        np.testing.assert_allclose(result.beta, np.ones(3), atol=1e-12)
        np.testing.assert_allclose(
            result.basis.conj().T @ result.basis,
            np.eye(4),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.basis.conj().T @ hamiltonian @ result.basis,
            result.tridiagonal,
            atol=1e-12,
        )

    def test_non_hermitian_operator_is_rejected(self):
        operator = np.array([[0.0, 1.0], [0.0, 0.0]])
        with self.assertRaises(ValueError):
            lanczos(operator, basis_state(0, 2))


class DynamicsAndMetricTests(unittest.TestCase):
    def test_pauli_x_spread_has_analytical_value(self):
        time = 0.37
        initial_state = basis_state(0, 2)
        result = lanczos(SIGMA_X, initial_state)
        state = exact_state(SIGMA_X, initial_state, time)
        probabilities = state_probabilities(state, result.basis)
        expected = np.array([np.cos(time) ** 2, np.sin(time) ** 2])
        np.testing.assert_allclose(probabilities, expected, atol=1e-12)
        self.assertAlmostEqual(spread_complexity(probabilities), np.sin(time) ** 2, places=12)
        self.assertAlmostEqual(
            participation_ratio(probabilities),
            1.0 / np.sum(expected**2),
            places=12,
        )
        expected_entropy = -np.sum(expected * np.log(expected))
        self.assertAlmostEqual(krylov_entropy(probabilities), expected_entropy, places=12)

    def test_full_krylov_projection_matches_exact_evolution(self):
        hamiltonian = path_hamiltonian(4)
        initial_state = basis_state(0, 4)
        result = lanczos(hamiltonian, initial_state)
        for time in (0.0, 0.2, 0.9):
            np.testing.assert_allclose(
                projected_state(result, time),
                exact_state(hamiltonian, initial_state, time),
                atol=1e-11,
            )


class ArnoldiTests(unittest.TestCase):
    def test_general_operator_projection(self):
        operator = np.array([[0.0, 1.0], [-1.0, 0.5j]], dtype=np.complex128)
        result = arnoldi(operator, basis_state(0, 2))
        np.testing.assert_allclose(
            result.basis.conj().T @ result.basis,
            np.eye(result.dimension),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.basis.conj().T @ operator @ result.basis,
            result.hessenberg,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
