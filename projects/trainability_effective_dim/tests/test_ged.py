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
        net = DummyNet(parameter_count=1)
        fishers = torch.tensor(
            [
                [[1.0]],
                [[1.0]],
            ],
            dtype=torch.float64,
        )

        sample_sizes = [10, 100, 1_000]

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