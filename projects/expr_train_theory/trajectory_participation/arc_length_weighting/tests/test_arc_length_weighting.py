import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation.arc_length_weighting import (
    arc_length_snapshot_weights,
    arc_length_weighted_participation_dimension,
    cumulative_fubini_study_length,
    fubini_study_distance,
)
from projects.expr_train_theory.trajectory_participation.sampling import (
    qubit_arc_states,
    weighted_trajectory_participation_dimension,
)


class ArcLengthWeightingTests(unittest.TestCase):
    def test_fubini_study_distance_known_qubit_arc(self):
        a = np.array([1.0, 0.0], dtype=complex)
        b = np.array([np.cos(0.3), np.sin(0.3)], dtype=complex)
        self.assertAlmostEqual(fubini_study_distance(a, b), 0.3, places=12)

    def test_distance_is_global_phase_invariant(self):
        a = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)
        b = np.array([0.6, 0.8], dtype=complex)
        base = fubini_study_distance(a, b)
        shifted = fubini_study_distance(np.exp(0.7j) * a, np.exp(-1.2j) * b)
        self.assertAlmostEqual(base, shifted, places=12)

    def test_cumulative_length_matches_qubit_arc_parameter(self):
        parameters = np.array([0.0, 0.1, 0.4, 0.9])
        cumulative = cumulative_fubini_study_length(qubit_arc_states(parameters))
        np.testing.assert_allclose(cumulative, parameters, atol=1e-12)

    def test_uniform_arc_sampling_gives_trapezoidal_weights(self):
        parameters = np.linspace(0.0, np.pi / 2.0, 5)
        weights = arc_length_snapshot_weights(qubit_arc_states(parameters))
        expected = np.array([0.125, 0.25, 0.25, 0.25, 0.125])
        np.testing.assert_allclose(weights, expected, atol=1e-12)

    def test_stationary_projective_path_falls_back_to_equal_weights(self):
        state = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)
        states = np.array([state, np.exp(0.2j) * state, np.exp(-0.8j) * state])
        weights = arc_length_snapshot_weights(states)
        np.testing.assert_allclose(weights, np.full(3, 1.0 / 3.0), atol=1e-12)
        self.assertAlmostEqual(arc_length_weighted_participation_dimension(states), 1.0, places=12)

    def test_nonuniform_sampling_is_close_to_dense_reference(self):
        dense = qubit_arc_states(np.linspace(0.0, np.pi / 2.0, 401))
        nonuniform = qubit_arc_states(
            np.array([0.0, 0.01, 0.03, 0.08, 0.25, 0.55, 0.95, 1.3, np.pi / 2.0])
        )
        dense_value = arc_length_weighted_participation_dimension(dense)
        nonuniform_value = arc_length_weighted_participation_dimension(nonuniform)
        equal_value = weighted_trajectory_participation_dimension(nonuniform)
        self.assertLess(abs(dense_value - nonuniform_value), 0.04)
        self.assertLess(
            abs(dense_value - nonuniform_value),
            abs(dense_value - equal_value),
        )

    def test_arc_weighted_dimension_is_phase_invariant(self):
        parameters = np.array([0.0, 0.2, 0.5, 1.0, np.pi / 2.0])
        states = qubit_arc_states(parameters)
        phases = np.exp(1j * np.array([0.1, -0.3, 0.7, 1.4, -0.9]))
        shifted = states * phases[:, None]
        self.assertAlmostEqual(
            arc_length_weighted_participation_dimension(states),
            arc_length_weighted_participation_dimension(shifted),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
