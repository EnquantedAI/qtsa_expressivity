import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation.calibration.cases import (
    duplicated_direction_case,
    identical_states,
    nearly_collinear_case,
    orthogonal_states,
)
from projects.expr_train_theory.trajectory_participation.calibration.metrics import calibrate_trajectory


class TrajectoryCalibrationTests(unittest.TestCase):
    def test_identical_states_hit_lower_bound(self):
        result = calibrate_trajectory(identical_states(count=4, dimension=4))
        self.assertAlmostEqual(result.dimension, 1.0, places=12)
        self.assertEqual(result.numerical_rank, 1)
        self.assertEqual(result.ceiling, 4)
        self.assertAlmostEqual(result.fraction_of_rank, 1.0, places=12)
        self.assertAlmostEqual(result.fraction_of_ceiling, 0.25, places=12)

    def test_orthogonal_states_hit_upper_bound(self):
        result = calibrate_trajectory(orthogonal_states(count=4, dimension=4))
        self.assertAlmostEqual(result.dimension, 4.0, places=12)
        self.assertEqual(result.numerical_rank, 4)
        self.assertAlmostEqual(result.fraction_of_rank, 1.0, places=12)
        self.assertAlmostEqual(result.fraction_of_ceiling, 1.0, places=12)

    def test_duplicated_direction_has_known_value(self):
        result = calibrate_trajectory(duplicated_direction_case())
        self.assertAlmostEqual(result.dimension, 9.0 / 5.0, places=12)
        self.assertEqual(result.numerical_rank, 2)
        self.assertAlmostEqual(result.fraction_of_rank, 0.9, places=12)
        self.assertAlmostEqual(result.fraction_of_ceiling, 0.6, places=12)

    def test_basic_bounds_hold(self):
        for states in [
            identical_states(),
            orthogonal_states(),
            duplicated_direction_case(),
            nearly_collinear_case(),
        ]:
            result = calibrate_trajectory(states)
            self.assertGreaterEqual(result.dimension, 1.0 - 1e-12)
            self.assertLessEqual(result.dimension, result.numerical_rank + 1e-10)
            self.assertLessEqual(result.numerical_rank, result.ceiling)
            self.assertGreaterEqual(result.fraction_of_rank, 0.0)
            self.assertLessEqual(result.fraction_of_rank, 1.0 + 1e-10)
            self.assertGreaterEqual(result.fraction_of_ceiling, 0.0)
            self.assertLessEqual(result.fraction_of_ceiling, 1.0 + 1e-10)

    def test_renyi2_dimension_does_not_exceed_entropy_dimension(self):
        result = calibrate_trajectory(duplicated_direction_case())
        self.assertLessEqual(result.dimension, result.entropy_dimension + 1e-12)
        self.assertLessEqual(result.entropy_dimension, result.numerical_rank + 1e-12)

    def test_stable_rank_is_bounded_by_dtp(self):
        result = calibrate_trajectory(duplicated_direction_case())
        self.assertLessEqual(result.stable_rank, result.dimension + 1e-12)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_trajectory(np.asarray([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
