import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.trajectory_participation.cross_metric_validation.study import (
    _mixed_axis_trajectory,
    _phase_trajectory,
    _same_axis_trajectory,
    compare_case,
    run_reference_cases,
)


class CrossMetricTests(unittest.TestCase):
    def test_same_axis_parameters_are_redundant_in_qfim(self):
        result = compare_case(
            "same-axis",
            np.array([0.4, -0.7]),
            _same_axis_trajectory,
        )
        self.assertEqual(result.parameter_count, 2)
        self.assertEqual(result.qfim_rank, 1)
        self.assertEqual(result.cfim_rank, 1)
        self.assertGreaterEqual(result.d_tp, 1.0)
        self.assertLessEqual(result.d_tp, 2.0 + 1e-10)

    def test_phase_parameter_is_invisible_to_computational_basis_cfim(self):
        result = compare_case("phase", np.array([0.8]), _phase_trajectory)
        self.assertEqual(result.qfim_rank, 1)
        self.assertEqual(result.cfim_rank, 0)
        self.assertGreater(result.d_tp, 1.0)

    def test_mixed_axis_case_is_bounded_by_qubit_hilbert_space(self):
        result = compare_case("mixed", np.array([0.7, 1.1]), _mixed_axis_trajectory)
        self.assertLessEqual(result.trajectory_rank, 2)
        self.assertLessEqual(result.d_tp, 2.0 + 1e-10)
        self.assertEqual(result.qfim_rank, 2)

    def test_reference_cases_write_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            results = run_reference_cases(directory)
            self.assertEqual(len(results), 3)
            self.assertTrue((Path(directory) / "cross_metric_reference.csv").exists())
            self.assertTrue((Path(directory) / "cross_metric_reference_metadata.json").exists())

    def test_bad_trajectory_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_case("bad", np.array([0.1]), lambda _: np.array([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
