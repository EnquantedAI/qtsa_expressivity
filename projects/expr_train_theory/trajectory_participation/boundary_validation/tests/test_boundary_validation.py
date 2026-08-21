import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation.boundary_validation.cases import (
    collapsed_trajectory_case,
    constant_output_qntk_case,
    linear_qntk_case,
    orthogonal_trajectory_case,
    redundant_parameter_qntk_case,
)


class BoundaryValidationTests(unittest.TestCase):
    def test_collapsed_trajectory_has_dimension_one(self):
        row = collapsed_trajectory_case()
        self.assertAlmostEqual(row["d_tp"], 1.0, places=10)
        self.assertEqual(row["rank"], 1)

    def test_orthogonal_trajectory_reaches_snapshot_count(self):
        row = orthogonal_trajectory_case(4)
        self.assertAlmostEqual(row["d_tp"], 4.0, places=10)
        self.assertEqual(row["rank"], 4)

    def test_constant_output_has_zero_qntk(self):
        row = constant_output_qntk_case()
        self.assertLess(row["jacobian_norm"], 1e-10)
        self.assertLess(row["kernel_norm"], 1e-10)
        self.assertEqual(row["rank"], 0)
        self.assertAlmostEqual(row["trace"], 0.0, places=12)

    def test_linear_model_matches_analytic_kernel(self):
        row = linear_qntk_case()
        self.assertLess(row["jacobian_error"], 1e-8)
        self.assertLess(row["kernel_error"], 1e-8)
        self.assertEqual(row["rank"], 2)

    def test_redundant_parameters_give_one_tangent_direction(self):
        row = redundant_parameter_qntk_case()
        self.assertEqual(row["parameter_count"], 2)
        self.assertEqual(row["jacobian_rank"], 1)
        self.assertEqual(row["qntk_rank"], 1)


if __name__ == "__main__":
    unittest.main()
