import unittest

import numpy as np

from projects.expr_train_theory.effective_dimension_checks.fisher_examples import (
    classical_fisher,
    computational_basis_probabilities,
    phase_state,
    ry_state,
)
from projects.expr_train_theory.qfim.core import compute_pure_state_qfim


def basis_probabilities(state_function):
    return lambda theta: computational_basis_probabilities(state_function(theta))


class TestFisherExamples(unittest.TestCase):
    def test_ry_measurement_reaches_qfim(self):
        theta = np.array([0.63])
        qfim = compute_pure_state_qfim(ry_state, theta)
        cfim = classical_fisher(basis_probabilities(ry_state), theta)
        np.testing.assert_allclose(qfim, [[1.0]], atol=1e-7)
        np.testing.assert_allclose(cfim, qfim, atol=1e-7)

    def test_computational_basis_does_not_see_phase_parameter(self):
        theta = np.array([0.63])
        qfim = compute_pure_state_qfim(phase_state, theta)
        cfim = classical_fisher(basis_probabilities(phase_state), theta)
        np.testing.assert_allclose(qfim, [[1.0]], atol=1e-7)
        np.testing.assert_allclose(cfim, [[0.0]], atol=1e-10)

    def test_cfim_does_not_exceed_qfim_in_examples(self):
        theta = np.array([0.31])
        for state_function in (ry_state, phase_state):
            qfim = compute_pure_state_qfim(state_function, theta)
            cfim = classical_fisher(basis_probabilities(state_function), theta)
            eigenvalues = np.linalg.eigvalsh(qfim - cfim)
            self.assertGreaterEqual(float(eigenvalues.min()), -1e-7)

    def test_zero_probability_point_needs_special_care(self):
        at_boundary = classical_fisher(
            basis_probabilities(ry_state),
            np.array([0.0]),
        )
        near_boundary = classical_fisher(
            basis_probabilities(ry_state),
            np.array([1e-4]),
        )
        np.testing.assert_allclose(at_boundary, [[0.0]], atol=1e-12)
        np.testing.assert_allclose(near_boundary, [[1.0]], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
