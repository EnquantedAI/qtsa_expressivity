import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation.core import (
    trajectory_gram,
    trajectory_participation_dimension,
    trajectory_participation_dimension_from_gram,
)
from projects.expr_train_theory.trajectory_participation.examples import (
    duplicated_orthogonal_states,
    identical_states,
    orthogonal_states,
)


class TrajectoryParticipationTests(unittest.TestCase):
    def test_identical_states_have_dimension_one(self):
        result = trajectory_participation_dimension(identical_states(count=5, dimension=3))
        self.assertAlmostEqual(result.dimension, 1.0, places=10)
        self.assertEqual(result.numerical_rank, 1)

    def test_orthogonal_states_reach_snapshot_count(self):
        states = orthogonal_states(count=4)
        result = trajectory_participation_dimension(states)
        self.assertAlmostEqual(result.dimension, 4.0, places=10)
        self.assertEqual(result.numerical_rank, 4)

    def test_duplicate_multiplicity_has_known_dimension(self):
        result = trajectory_participation_dimension(duplicated_orthogonal_states())
        self.assertAlmostEqual(result.dimension, 8.0 / 3.0, places=10)

    def test_independent_global_phases_do_not_change_dimension(self):
        states = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=complex,
        )
        phases = np.exp(1j * np.array([0.3, -1.1, 2.2]))[:, None]
        base = trajectory_participation_dimension(states)
        shifted = trajectory_participation_dimension(states * phases)
        self.assertAlmostEqual(base.dimension, shifted.dimension, places=10)
        np.testing.assert_allclose(base.spectrum, shifted.spectrum, atol=1e-10)

    def test_unitary_change_of_basis_does_not_change_dimension(self):
        rng = np.random.default_rng(123)
        states = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        rotated = states @ q.T
        base = trajectory_participation_dimension(states)
        changed = trajectory_participation_dimension(rotated)
        self.assertAlmostEqual(base.dimension, changed.dimension, places=10)
        np.testing.assert_allclose(base.spectrum, changed.spectrum, atol=1e-10)

    def test_dimension_is_bounded_by_rank_and_snapshot_count(self):
        rng = np.random.default_rng(7)
        states = rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3))
        result = trajectory_participation_dimension(states)
        self.assertGreaterEqual(result.dimension, 1.0 - 1e-10)
        self.assertLessEqual(result.dimension, min(6, 3) + 1e-10)

    def test_gram_has_unit_diagonal_after_normalization(self):
        states = np.array([[2.0, 0.0], [1.0, 1.0]], dtype=complex)
        gram = trajectory_gram(states)
        np.testing.assert_allclose(np.diag(gram), np.ones(2), atol=1e-10)

    def test_gram_and_svd_forms_agree(self):
        rng = np.random.default_rng(19)
        states = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
        svd_value = trajectory_participation_dimension(states).dimension
        gram_value = trajectory_participation_dimension_from_gram(states)
        self.assertAlmostEqual(svd_value, gram_value, places=10)

    def test_zero_state_is_rejected(self):
        with self.assertRaises(ValueError):
            trajectory_participation_dimension([[1.0, 0.0], [0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
