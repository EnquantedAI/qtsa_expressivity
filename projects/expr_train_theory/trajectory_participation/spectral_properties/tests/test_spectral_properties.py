import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation.core import (
    trajectory_participation_dimension,
)
from projects.expr_train_theory.trajectory_participation.spectral_properties.metrics import (
    renyi_effective_dimension,
    renyi_entropy,
    spectral_dimension_profile,
)


class SpectralPropertiesTests(unittest.TestCase):
    def test_participation_dimension_is_renyi_two_effective_dimension(self):
        states = np.array([[1, 0], [1, 0], [0, 1]], dtype=complex)
        result = trajectory_participation_dimension(states)
        self.assertAlmostEqual(
            renyi_effective_dimension(result.spectrum, 2.0),
            result.dimension,
            places=12,
        )

    def test_uniform_spectrum_gives_rank_for_all_orders(self):
        spectrum = np.full(4, 0.25)
        for order in (1.0, 2.0, 3.0, np.inf):
            self.assertAlmostEqual(renyi_effective_dimension(spectrum, order), 4.0)

    def test_dimension_hierarchy(self):
        states = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=complex)
        p = spectral_dimension_profile(states)
        self.assertLessEqual(p.stable_rank, p.participation_dimension + 1e-12)
        self.assertLessEqual(p.participation_dimension, p.entropy_dimension + 1e-12)
        self.assertLessEqual(p.entropy_dimension, p.numerical_rank + 1e-12)

    def test_identical_states_collapse_all_dimensions_to_one(self):
        states = np.tile(np.array([[1.0, 0.0]], dtype=complex), (5, 1))
        p = spectral_dimension_profile(states)
        self.assertEqual(p.numerical_rank, 1)
        self.assertAlmostEqual(p.stable_rank, 1.0)
        self.assertAlmostEqual(p.participation_dimension, 1.0)
        self.assertAlmostEqual(p.entropy_dimension, 1.0)

    def test_weighted_profile_uses_weighted_spectrum(self):
        states = np.eye(2, dtype=complex)
        p = spectral_dimension_profile(states, weights=[0.9, 0.1])
        expected = 1.0 / (0.9**2 + 0.1**2)
        self.assertAlmostEqual(p.participation_dimension, expected)
        self.assertEqual(p.numerical_rank, 2)

    def test_invalid_renyi_order_is_rejected(self):
        with self.assertRaises(ValueError):
            renyi_entropy([0.5, 0.5], 0.0)


if __name__ == "__main__":
    unittest.main()
