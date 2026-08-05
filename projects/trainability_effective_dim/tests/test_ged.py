import math
import unittest
from unittest.mock import patch

import torch

from projects.trainability_effective_dim.core.measures.ged import (
    estimate_global_effective_dimension,
)


class DummyQuantumLayer:
    def __init__(self, weights):
        self.weights = weights


class DummyNet:
    def __init__(self, parameter_count):
        self.parameter_count = parameter_count
        self.qlayers = DummyQuantumLayer(
            torch.zeros(parameter_count, dtype=torch.float64)
        )


class TestGlobalEffectiveDimension(unittest.TestCase):
    def test_one_dimensional_constant_fisher_has_known_ged(self):
        """
        Verify GED against its analytic value for a constant one-dimensional
        empirical Fisher matrix.

        Every sampled parameter vector has Fisher matrix [[2]]. Because the
        mean Fisher trace is also 2 and d = 1, every normalized Fisher matrix
        is F_hat = [[1]]. The Monte Carlo average is therefore exact.

        Expected result:
            - GED equals log(1 + kappa) / log(kappa).
            - The result contains exactly one effective-dimension value.
        """
        net = DummyNet(parameter_count=1)
        fishers = torch.tensor(
            [
                [[2.0]],
                [[2.0]],
                [[2.0]],
            ],
            dtype=torch.float64,
        )

        n = 100
        kappa = n / (2.0 * math.pi * math.log(n))
        expected = math.log1p(kappa) / math.log(kappa)

        with patch(
            "projects.trainability_effective_dim.core.measures.ged."
            "sample_empirical_fishers",
            return_value=fishers,
        ):
            actual = estimate_global_effective_dimension(
                net=net,
                inputs=torch.zeros((2, 1), dtype=torch.float64),
                n_theta=3,
                array_of_theoretical_number_of_data_samples=[n],
            )

        self.assertEqual(len(actual), 1)
        self.assertAlmostEqual(actual[0], expected, places=12)

    def test_ged_uses_each_parameter_sample_before_averaging(self):
        """
        Verify that GED computes the determinant contribution separately for
        every sampled parameter vector before the final Monte Carlo average.

        The supplied Fisher matrices are [[1]] and [[3]]. Their mean trace is
        2, giving normalized values [[0.5]] and [[1.5]]. GED must average
        sqrt(1 + kappa * F_hat) over these two samples, rather than calculate
        the determinant after prematurely averaging the Fisher matrices.

        Expected result:
            - GED equals the explicit two-sample Monte Carlo expression.
            - The test fails if Fisher matrices are averaged before their
              individual log-determinants are computed.
        """
        net = DummyNet(parameter_count=1)
        fishers = torch.tensor(
            [
                [[1.0]],
                [[3.0]],
            ],
            dtype=torch.float64,
        )

        n = 100
        kappa = n / (2.0 * math.pi * math.log(n))

        mean_fisher_trace = 2.0
        f_hat_0 = 1.0 / mean_fisher_trace
        f_hat_1 = 3.0 / mean_fisher_trace

        expected = (
            2.0
            * math.log(
                (
                    math.sqrt(1.0 + kappa * f_hat_0)
                    + math.sqrt(1.0 + kappa * f_hat_1)
                )
                / 2.0
            )
            / math.log(kappa)
        )

        with patch(
            "projects.trainability_effective_dim.core.measures.ged."
            "sample_empirical_fishers",
            return_value=fishers,
        ):
            actual = estimate_global_effective_dimension(
                net=net,
                inputs=torch.zeros((2, 1), dtype=torch.float64),
                n_theta=2,
                array_of_theoretical_number_of_data_samples=[n],
            )

        self.assertAlmostEqual(actual[0], expected, places=12)

    def test_ged_is_invariant_under_global_fisher_scaling(self):
        """
        Verify invariance of LED under multiplication of all Fisher matrices
        by the same positive scalar.

        LED normalizes each Fisher matrix using the trace of the mean Fisher
        matrix. Therefore, multiplying every Fisher matrix by 12 multiplies
        numerator and normalization trace equally, leaving F_hat unchanged.

        Expected result:
            - LED from `fishers` equals LED from `12 * fishers`.
            - This validates the Fisher-trace normalization.
        """
        net = DummyNet(parameter_count=2)

        fishers = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 1.0]],
                [[4.0, 0.0], [0.0, 3.0]],
            ],
            dtype=torch.float64,
        )

        scaled_fishers = 7.5 * fishers
        n = 100

        with patch(
            "projects.trainability_effective_dim.core.measures.ged."
            "sample_empirical_fishers",
            return_value=fishers,
        ):
            unscaled_result = estimate_global_effective_dimension(
                net=net,
                inputs=torch.zeros((2, 1), dtype=torch.float64),
                n_theta=2,
                array_of_theoretical_number_of_data_samples=[n],
            )

        with patch(
            "projects.trainability_effective_dim.core.measures.ged."
            "sample_empirical_fishers",
            return_value=scaled_fishers,
        ):
            scaled_result = estimate_global_effective_dimension(
                net=net,
                inputs=torch.zeros((2, 1), dtype=torch.float64),
                n_theta=2,
                array_of_theoretical_number_of_data_samples=[n],
            )

        self.assertAlmostEqual(
            unscaled_result[0],
            scaled_result[0],
            places=12,
        )

    def test_returns_one_value_per_theoretical_dataset_size(self):
        """
        Verify that GED returns one finite value for every requested
        theoretical dataset size.

        For constant one-dimensional Fisher matrices, the normalized Fisher
        is [[1]] for every parameter sample. Dataset sizes are selected such
        that kappa > 1, avoiding the invalid negative-log-kappa regime.

        Expected result:
            - The returned list has the same length as `sample_sizes`.
            - Each GED value is finite.
        """
        net = DummyNet(parameter_count=1)
        fishers = torch.tensor(
            [
                [[1.0]],
                [[1.0]],
            ],
            dtype=torch.float64,
        )

        sample_sizes = [100, 1_000, 10_000]

        with patch(
            "projects.trainability_effective_dim.core.measures.ged."
            "sample_empirical_fishers",
            return_value=fishers,
        ):
            actual = estimate_global_effective_dimension(
                net=net,
                inputs=torch.zeros((2, 1), dtype=torch.float64),
                n_theta=2,
                array_of_theoretical_number_of_data_samples=sample_sizes,
            )

        self.assertEqual(len(actual), len(sample_sizes))
        self.assertTrue(all(math.isfinite(value) for value in actual))

    def test_rejects_dataset_size_at_most_one(self):
        """
        Verify that GED rejects an invalid theoretical dataset size.

        GED uses log(n) to define kappa. This is undefined for n = 1, so the
        estimator must raise ValueError before attempting the calculation.

        Expected result:
            - Calling GED with n = 1 raises ValueError.
        """
        net = DummyNet(parameter_count=1)
        fishers = torch.tensor([[[1.0]]], dtype=torch.float64)

        with patch(
            "projects.trainability_effective_dim.core.measures.ged."
            "sample_empirical_fishers",
            return_value=fishers,
        ):
            with self.assertRaises(ValueError):
                estimate_global_effective_dimension(
                    net=net,
                    inputs=torch.zeros((2, 1), dtype=torch.float64),
                    n_theta=1,
                    array_of_theoretical_number_of_data_samples=[1],
                )


if __name__ == "__main__":
    unittest.main()