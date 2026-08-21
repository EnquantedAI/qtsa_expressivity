import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation import (
    trajectory_participation_dimension,
)
from projects.expr_train_theory.trajectory_participation.gram_backend import (
    analyse_trajectory_from_gram,
    gram_matrix_from_states,
    participation_dimension_from_gram_matrix,
    spectrum_from_gram_matrix,
)


class GramBackendTests(unittest.TestCase):
    def test_gram_backend_matches_svd_backend(self):
        rng = np.random.default_rng(11)
        states = rng.normal(size=(5, 16)) + 1j * rng.normal(size=(5, 16))
        svd_result = trajectory_participation_dimension(states)
        gram_result = analyse_trajectory_from_gram(gram_matrix_from_states(states))
        self.assertAlmostEqual(gram_result.dimension, svd_result.dimension, places=11)
        self.assertEqual(gram_result.numerical_rank, svd_result.numerical_rank)
        np.testing.assert_allclose(
            gram_result.spectrum,
            np.sort(svd_result.spectrum)[::-1],
            atol=1e-11,
        )

    def test_orthogonal_states_have_maximum_dimension(self):
        states = np.eye(4, dtype=complex)
        gram = gram_matrix_from_states(states)
        self.assertAlmostEqual(participation_dimension_from_gram_matrix(gram), 4.0)

    def test_identical_states_have_dimension_one(self):
        states = np.tile(np.array([[1.0, 0.0]], dtype=complex), (6, 1))
        gram = gram_matrix_from_states(states)
        self.assertAlmostEqual(participation_dimension_from_gram_matrix(gram), 1.0)

    def test_gram_spectrum_matches_known_duplicate_case(self):
        states = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=complex)
        spectrum = spectrum_from_gram_matrix(gram_matrix_from_states(states))
        np.testing.assert_allclose(spectrum, [2.0 / 3.0, 1.0 / 3.0], atol=1e-12)

    def test_precomputed_gram_does_not_need_statevectors(self):
        gram = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=complex)
        expected = 4.0 / 2.5
        self.assertAlmostEqual(participation_dimension_from_gram_matrix(gram), expected)

    def test_non_hermitian_matrix_is_rejected(self):
        gram = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=complex)
        with self.assertRaises(ValueError):
            participation_dimension_from_gram_matrix(gram)

    def test_non_psd_matrix_is_rejected(self):
        gram = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=complex)
        with self.assertRaises(ValueError):
            participation_dimension_from_gram_matrix(gram)


if __name__ == "__main__":
    unittest.main()
