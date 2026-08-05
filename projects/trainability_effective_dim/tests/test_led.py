import math
import unittest
from unittest.mock import patch

import torch

from projects.trainability_effective_dim.core.measures.led import (
    estimate_local_effective_dimension,
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


class TestLocalEffectiveDimension(unittest.TestCase):
    def test_constant_one_dimensional_fisher_has_known_led(self):
        """
        Verify LED against its analytic value for a constant one-dimensional
        empirical Fisher matrix.

        Each sampled parameter point has Fisher matrix [[2]]. Since the mean
        Fisher trace is also 2 and d = 1, the normalized Fisher matrix is
        F_hat = [[1]] for every parameter sample. The Monte Carlo average is
        therefore exact.

        Expected result:
            - LED equals log(1 + kappa) / log(kappa).
            - The returned result contains exactly one value.
            - The epsilon value does not affect this mocked Fisher input.
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
            "projects.trainability_effective_dim.core.measures.led."
            "sample_empirical_fishers",
            return_value=fishers,
        ):
            actual = estimate_local_effective_dimension(
                net=net,
                inputs=torch.zeros((2, 1), dtype=torch.float64),
                epsilon=0.2,
                n_theta=3,
                array_of_theoretical_number_of_data_samples=[n],
            )

        self.assertEqual(len(actual), 1)
        self.assertAlmostEqual(actual[0], expected, places=12)

    def test_led_uses_each_parameter_sample_before_averaging(self):
        """
        Verify that LED evaluates the log-determinant separately for every
        sampled parameter vector before performing the Monte Carlo average.

        The two Fisher matrices are [[1]] and [[3]]. Their mean trace is 2,
        so the normalized matrices are [[0.5]] and [[1.5]]. LED must average
        sqrt(1 + kappa * F_hat) values, not compute the determinant from a
        Fisher matrix averaged before this nonlinear operation.

        Expected result:
            - LED equals the explicit two-sample Monte Carlo expression.
            - The test fails if Fisher matrices are averaged before computing
              individual log-determinants.
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

        expected = (
            2.0
            * math.log(
                (
                    math.sqrt(1.0 + kappa * 0.5)
                    + math.sqrt(1.0 + kappa * 1.5)
                )
                / 2.0
            )
            / math.log(kappa)
        )

        with patch(
            "projects.trainability_effective_dim.core.measures.led."
            "sample_empirical_fishers",
            return_value=fishers,
        ):
            actual = estimate_local_effective_dimension(
                net=net,
                inputs=torch.zeros((2, 1), dtype=torch.float64),
                epsilon=0.1,
                n_theta=2,
                array_of_theoretical_number_of_data_samples=[n],
            )

        self.assertAlmostEqual(actual[0], expected, places=12)

    def test_led_is_invariant_under_global_fisher_scaling(self):
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

        with patch(
            "projects.trainability_effective_dim.core.measures.led."
            "sample_empirical_fishers",
            return_value=fishers,
        ):
            original_result = estimate_local_effective_dimension(
                net=net,
                inputs=torch.zeros((2, 1), dtype=torch.float64),
                n_theta=2,
                array_of_theoretical_number_of_data_samples=[100],
            )

        with patch(
            "projects.trainability_effective_dim.core.measures.led."
            "sample_empirical_fishers",
            return_value=12.0 * fishers,
        ):
            scaled_result = estimate_local_effective_dimension(
                net=net,
                inputs=torch.zeros((2, 1), dtype=torch.float64),
                n_theta=2,
                array_of_theoretical_number_of_data_samples=[100],
            )

        self.assertAlmostEqual(
            original_result[0],
            scaled_result[0],
            places=12,
        )

    def test_passes_epsilon_ball_sampler_to_monte_carlo(self):
        """
        Verify that LED constructs and passes an epsilon-ball parameter-space
        sampler to the empirical-Fisher Monte Carlo estimator.

        LED is local because it samples parameter values near the current
        model weights, rather than sampling from a global parameter domain.

        Expected result:
            - The Monte Carlo estimator receives an EpsilonBallParameterSpace.
            - The sampler epsilon equals the epsilon supplied to LED.
            - The sampler centre equals the network's current weights.
        """
        net = DummyNet(parameter_count=2)
        fishers = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
            dtype=torch.float64,
        )

        epsilon = 0.37

        with patch(
            "projects.trainability_effective_dim.core.measures.led."
            "sample_empirical_fishers",
            return_value=fishers,
        ) as mocked_sampler:
            estimate_local_effective_dimension(
                net=net,
                inputs=torch.zeros((2, 1), dtype=torch.float64),
                epsilon=epsilon,
                n_theta=2,
                array_of_theoretical_number_of_data_samples=[100],
            )

        parameter_space = mocked_sampler.call_args.kwargs["parameter_space"]

        self.assertEqual(parameter_space.epsilon, epsilon)

        torch.testing.assert_close(
            parameter_space.center,
            net.qlayers.weights,
            atol=0.0,
            rtol=0.0,
        )

    def test_rejects_dataset_size_at_most_one(self):
        """
        Verify rejection of an invalid theoretical dataset size.

        The LED normalization uses log(n), which is undefined at n = 1 and
        inappropriate for n below or equal to one.

        Expected result:
            - Calling LED with n = 1 raises ValueError.
        """
        net = DummyNet(parameter_count=1)
        fishers = torch.tensor([[[1.0]]], dtype=torch.float64)

        with patch(
            "projects.trainability_effective_dim.core.measures.led."
            "sample_empirical_fishers",
            return_value=fishers,
        ):
            with self.assertRaises(ValueError):
                estimate_local_effective_dimension(
                    net=net,
                    inputs=torch.zeros((2, 1), dtype=torch.float64),
                    n_theta=1,
                    array_of_theoretical_number_of_data_samples=[1],
                )


if __name__ == "__main__":
    unittest.main()