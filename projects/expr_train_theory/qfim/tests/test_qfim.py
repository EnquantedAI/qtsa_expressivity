"""Analytical checks for the pure-state QFIM."""

from __future__ import annotations

import unittest

import numpy as np

from projects.expr_train_theory.qfim.core import compute_pure_state_qfim, diagnose_qfim
from projects.expr_train_theory.qfim.validation_models import (
    global_phase_state,
    product_ry_state,
    redundant_ry_state,
    ry_state,
)


class PureStateQFIMTests(unittest.TestCase):
    def assertMatrixClose(self, actual: np.ndarray, expected: np.ndarray, atol: float = 2e-6) -> None:
        self.assertTrue(
            np.allclose(actual, expected, atol=atol, rtol=atol),
            msg=f"\nactual=\n{actual}\nexpected=\n{expected}",
        )

    def test_single_ry_has_unit_qfi(self) -> None:
        qfim = compute_pure_state_qfim(ry_state, np.array([0.37]))
        self.assertMatrixClose(qfim, np.array([[1.0]]))
        diagnostics = diagnose_qfim(qfim)
        self.assertEqual(diagnostics.numerical_rank, 1)
        self.assertAlmostEqual(diagnostics.relative_rank, 1.0)

    def test_redundant_parameters_give_rank_one(self) -> None:
        qfim = compute_pure_state_qfim(redundant_ry_state, np.array([0.2, -0.6]))
        self.assertMatrixClose(qfim, np.ones((2, 2)))
        diagnostics = diagnose_qfim(qfim)
        self.assertEqual(diagnostics.numerical_rank, 1)
        self.assertAlmostEqual(diagnostics.relative_rank, 0.5)
        self.assertMatrixClose(diagnostics.eigenvalues, np.array([2.0, 0.0]))

    def test_independent_product_rotations_give_identity(self) -> None:
        qfim = compute_pure_state_qfim(product_ry_state, np.array([0.2, 1.1]))
        self.assertMatrixClose(qfim, np.eye(2))
        self.assertEqual(diagnose_qfim(qfim).numerical_rank, 2)

    def test_global_phase_is_removed(self) -> None:
        qfim = compute_pure_state_qfim(global_phase_state, np.array([0.8]))
        self.assertMatrixClose(qfim, np.zeros((1, 1)), atol=5e-7)
        self.assertEqual(diagnose_qfim(qfim).numerical_rank, 0)

    def test_rejects_non_normalized_state(self) -> None:
        def invalid_state(_: np.ndarray) -> np.ndarray:
            return np.array([2.0, 0.0])

        with self.assertRaises(ValueError):
            compute_pure_state_qfim(invalid_state, np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
