import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation.core import trajectory_participation_dimension
from projects.expr_train_theory.trajectory_participation.sampling import (
    qubit_arc_states,
    trapezoidal_snapshot_weights,
    weighted_trajectory_participation_dimension,
)
from projects.expr_train_theory.trajectory_participation.sampling_sensitivity.study import (
    duplication_study,
    sampling_density_study,
)


class SamplingSensitivityTests(unittest.TestCase):
    def test_uniform_weights_match_equal_weight_definition(self):
        states = qubit_arc_states(np.linspace(0.0, 1.0, 7))
        equal = trajectory_participation_dimension(states).dimension
        weighted = weighted_trajectory_participation_dimension(states, np.ones(7))
        self.assertAlmostEqual(equal, weighted, places=10)

    def test_duplicate_state_changes_equal_weight_dimension(self):
        result = duplication_study(duplicate_count=8)
        self.assertNotAlmostEqual(result["base"], result["duplicated_equal"], places=6)

    def test_weights_can_restore_original_mixture_after_duplication(self):
        result = duplication_study(duplicate_count=8)
        self.assertAlmostEqual(result["base"], result["duplicated_weighted"], places=10)

    def test_trapezoidal_weights_are_normalized(self):
        x = np.array([0.0, 0.1, 0.5, 1.0])
        weights = trapezoidal_snapshot_weights(x)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=12)
        self.assertTrue(np.all(weights > 0.0))

    def test_density_study_accepts_multiple_sample_counts(self):
        rows = sampling_density_study(sample_counts=(3, 5, 9))
        self.assertEqual([row["sample_count"] for row in rows], [3, 5, 9])
        self.assertTrue(all(1.0 <= row["d_tp_equal"] <= 2.0 for row in rows))
        self.assertTrue(all(1.0 <= row["d_tp_trapezoidal"] <= 2.0 for row in rows))

    def test_negative_weights_are_rejected(self):
        states = np.eye(2, dtype=complex)
        with self.assertRaises(ValueError):
            weighted_trajectory_participation_dimension(states, [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
