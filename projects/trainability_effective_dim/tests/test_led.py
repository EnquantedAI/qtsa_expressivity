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