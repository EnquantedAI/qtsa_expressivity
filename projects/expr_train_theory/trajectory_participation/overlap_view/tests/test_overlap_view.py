import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation.core import (
    trajectory_participation_dimension,
)
from projects.expr_train_theory.trajectory_participation.overlap_view.metrics import (
    first_state_frame_potential,
    trajectory_density_purity_from_overlaps,
    trajectory_participation_dimension_from_overlaps,
    weighted_state_frame_potential,
)
from projects.expr_train_theory.trajectory_participation.sampling import (
    weighted_trajectory_participation_dimension,
)


class OverlapViewTests(unittest.TestCase):
    def test_equal_weight_overlap_formula_matches_svd_definition(self):
        states = np.array(
            [
                [1.0, 0.0],
                [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
                [0.0, 1.0],
            ],
            dtype=complex,
        )
        direct = trajectory_participation_dimension(states).dimension
        overlap = trajectory_participation_dimension_from_overlaps(states)
        self.assertAlmostEqual(direct, overlap, places=12)

    def test_orthogonal_states_have_minimal_frame_potential(self):
        states = np.eye(4, dtype=complex)
        self.assertAlmostEqual(first_state_frame_potential(states), 4.0)
        self.assertAlmostEqual(trajectory_participation_dimension_from_overlaps(states), 4.0)

    def test_identical_states_have_maximal_equal_weight_frame_potential(self):
        state = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)
        states = np.repeat(state[None, :], 5, axis=0)
        self.assertAlmostEqual(first_state_frame_potential(states), 25.0)
        self.assertAlmostEqual(trajectory_density_purity_from_overlaps(states), 1.0)
        self.assertAlmostEqual(trajectory_participation_dimension_from_overlaps(states), 1.0)

    def test_weighted_overlap_formula_matches_weighted_svd_definition(self):
        states = np.array(
            [
                [1.0, 0.0],
                [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
                [0.0, 1.0],
            ],
            dtype=complex,
        )
        weights = np.array([0.2, 0.3, 0.5])
        direct = weighted_trajectory_participation_dimension(states, weights)
        overlap = trajectory_participation_dimension_from_overlaps(states, weights)
        self.assertAlmostEqual(direct, overlap, places=12)

    def test_weighted_frame_potential_is_invariant_to_weight_rescaling(self):
        states = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        a = weighted_state_frame_potential(states, [1.0, 3.0])
        b = weighted_state_frame_potential(states, [10.0, 30.0])
        self.assertAlmostEqual(a, b, places=12)


if __name__ == "__main__":
    unittest.main()
