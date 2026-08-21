import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.trajectory_participation.qntk_cross_metric.qntk import (
    compute_qntk,
    diagnose_qntk,
    finite_difference_jacobian,
    qntk_from_jacobian,
)
from projects.expr_train_theory.trajectory_participation.qntk_cross_metric.study import (
    ToyCase,
    evaluate_case,
    save_results,
    toy_output,
)


class QNTKReferenceTests(unittest.TestCase):
    def test_linear_model_has_known_jacobian(self):
        X = np.array([-1.0, 0.0, 2.0])
        theta = np.array([0.3, -0.7])

        def model(x, parameters):
            return parameters[0] * x + parameters[1]

        jac = finite_difference_jacobian(model, theta, X)
        expected = np.column_stack([X, np.ones_like(X)])
        np.testing.assert_allclose(jac, expected, atol=1e-8)

    def test_qntk_is_j_j_transpose(self):
        jac = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(qntk_from_jacobian(jac), jac @ jac.T)

    def test_constant_model_has_zero_kernel(self):
        kernel = compute_qntk(lambda x, theta: 2.0, np.array([0.2, 0.4]), [0, 1, 2])
        np.testing.assert_allclose(kernel, 0.0)
        diag = diagnose_qntk(kernel)
        self.assertEqual(diag.rank, 0)
        self.assertEqual(diag.effective_rank, 0.0)

    def test_one_parameter_kernel_has_rank_at_most_one(self):
        X = np.linspace(-1.0, 1.0, 6)

        def model(x, theta):
            return np.cos(x + theta[0])

        diag = diagnose_qntk(compute_qntk(model, np.array([0.3]), X))
        self.assertEqual(diag.rank, 1)
        self.assertAlmostEqual(diag.effective_rank, 1.0, places=7)

    def test_phase_parameter_is_invisible_to_z_output(self):
        case_a = ToyCase("a", np.linspace(-1.0, 1.0, 5), np.array([0.3, 0.0]))
        case_b = ToyCase("b", np.linspace(-1.0, 1.0, 5), np.array([0.3, 1.5]))
        a = evaluate_case(case_a)
        b = evaluate_case(case_b)
        self.assertEqual(a["qntk_rank"], 1)
        self.assertEqual(b["qntk_rank"], 1)
        self.assertAlmostEqual(a["qntk_trace"], b["qntk_trace"], places=6)
        self.assertNotAlmostEqual(a["d_tp_mean"], b["d_tp_mean"], places=4)

    def test_toy_output_is_bounded(self):
        for x in np.linspace(-2.0, 2.0, 5):
            value = toy_output(x, np.array([0.5, 0.7]))
            self.assertLessEqual(abs(value), 1.0 + 1e-12)

    def test_save_results(self):
        rows = [{"case": "x", "d_tp_mean": 1.0}]
        with tempfile.TemporaryDirectory() as tmp:
            save_results(tmp, rows, {"seed": 1})
            names = {p.name for p in Path(tmp).iterdir()}
        self.assertIn("qntk_cross_metric.csv", names)
        self.assertIn("qntk_cross_metric_metadata.json", names)


if __name__ == "__main__":
    unittest.main()
