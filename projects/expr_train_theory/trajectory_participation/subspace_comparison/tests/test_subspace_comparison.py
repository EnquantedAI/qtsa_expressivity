import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation.subspace_comparison import (
    compare_hard_and_soft_dimension,
    modified_gram_schmidt,
)


class SubspaceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.e0 = np.array([1.0, 0.0])
        self.e1 = np.array([0.0, 1.0])

    def test_gram_schmidt_recovers_orthogonal_basis(self):
        basis, residuals = modified_gram_schmidt([self.e0, self.e1])
        self.assertEqual(basis.shape, (2, 2))
        np.testing.assert_allclose(basis @ basis.conj().T, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(residuals, [1.0, 1.0], atol=1e-12)

    def test_duplicate_snapshot_does_not_increase_hard_rank(self):
        result = compare_hard_and_soft_dimension([self.e0, self.e0, self.e1])
        self.assertEqual(result.gram_schmidt_rank, 2)
        self.assertEqual(result.svd_rank, 2)

    def test_balanced_orthogonal_trajectory_uses_full_rank(self):
        result = compare_hard_and_soft_dimension([self.e0, self.e1])
        self.assertAlmostEqual(result.participation_dimension, 2.0, places=12)
        self.assertAlmostEqual(result.rank_utilization, 1.0, places=12)

    def test_same_subspace_can_have_smaller_participation_dimension(self):
        balanced = compare_hard_and_soft_dimension([self.e0, self.e1])
        biased = compare_hard_and_soft_dimension([self.e0, self.e0, self.e0, self.e1])
        self.assertEqual(balanced.gram_schmidt_rank, biased.gram_schmidt_rank)
        self.assertAlmostEqual(biased.participation_dimension, 8.0 / 5.0, places=12)
        self.assertLess(biased.participation_dimension, balanced.participation_dimension)

    def test_tolerance_controls_nearly_dependent_direction(self):
        eps = 1e-8
        near = np.array([1.0, eps])
        strict = compare_hard_and_soft_dimension([self.e0, near], tol=1e-10)
        loose = compare_hard_and_soft_dimension([self.e0, near], tol=1e-6)
        self.assertEqual(strict.gram_schmidt_rank, 2)
        self.assertEqual(loose.gram_schmidt_rank, 1)
        self.assertEqual(strict.svd_rank, 2)
        self.assertEqual(loose.svd_rank, 1)

    def test_nonpositive_tolerance_is_rejected(self):
        with self.assertRaises(ValueError):
            modified_gram_schmidt([self.e0], tol=0.0)


if __name__ == "__main__":
    unittest.main()
