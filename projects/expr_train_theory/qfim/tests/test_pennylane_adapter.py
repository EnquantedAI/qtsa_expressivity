"""Tests for the PennyLane adapter."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from projects.expr_train_theory.qfim.core import (
    compute_pure_state_qfim,
    diagnose_qfim,
)

PENNYLANE_AVAILABLE = importlib.util.find_spec("pennylane") is not None


@unittest.skipUnless(PENNYLANE_AVAILABLE, "PennyLane is required for these tests")
class PennyLaneAdapterTests(unittest.TestCase):
    def test_shared_architecture_returns_normalized_state_and_square_qfim(self):
        from projects.expr_train_theory.qfim.pennylane_adapter import (
            build_shared_architecture_state_model,
        )

        model = build_shared_architecture_state_model(
            [0.2, 0.7], n_layers=1, feature_map="zzfm"
        )
        theta = np.linspace(-0.4, 0.5, model.parameter_count)
        state = model.state_function(theta)
        qfim = compute_pure_state_qfim(model.state_function, theta)
        diagnostics = diagnose_qfim(qfim)

        self.assertEqual(state.shape, (2**model.n_qubits,))
        self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=9)
        self.assertEqual(qfim.shape, (model.parameter_count, model.parameter_count))
        self.assertLess(diagnostics.symmetry_error, 1e-9)
        self.assertGreaterEqual(diagnostics.minimum_eigenvalue, -1e-5)

    def test_parameter_count_matches_strongly_entangling_shape(self):
        from projects.expr_train_theory.qfim.pennylane_adapter import (
            build_shared_architecture_state_model,
        )

        model = build_shared_architecture_state_model(
            [0.1, 0.2, 0.3], n_layers=2, feature_map="iqp"
        )
        self.assertEqual(model.weight_shape, (2, 3, 3))
        self.assertEqual(model.parameter_count, 18)

    def test_rejects_parameter_vector_with_wrong_size(self):
        from projects.expr_train_theory.qfim.pennylane_adapter import (
            build_shared_architecture_state_model,
        )

        model = build_shared_architecture_state_model(
            [0.1, 0.2], n_layers=1, feature_map="Y"
        )
        with self.assertRaises(ValueError):
            model.state_function(np.zeros(model.parameter_count + 1))


if __name__ == "__main__":
    unittest.main()
