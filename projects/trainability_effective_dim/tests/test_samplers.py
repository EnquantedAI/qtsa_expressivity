import unittest
import torch

from projects.trainability_effective_dim.core.measures.samplers import (
    UniformParameterSpace,
    EpsilonBallParameterSpace,
)


class TestSamplers(unittest.TestCase):
    def test_uniform_sample_has_expected_shape_and_dtype(self):
        sampler = UniformParameterSpace(
            low=-torch.pi,
            high=torch.pi,
            shape=(3, 2, 4),
            dtype=torch.float64,
        )

        sample = sampler.sample()

        self.assertEqual(sample.shape, torch.Size([3, 2, 4]))
        self.assertEqual(sample.dtype, torch.float64)

    def test_uniform_sample_lies_within_bounds(self):
        sampler = UniformParameterSpace(
            low=-2.0,
            high=3.0,
            shape=(1000,),
            dtype=torch.float64,
        )

        sample = sampler.sample()

        self.assertTrue(torch.all(sample >= -2.0))
        self.assertTrue(torch.all(sample < 3.0))

    def test_epsilon_ball_sample_has_expected_shape_and_dtype(self):
        center = torch.zeros((2, 3), dtype=torch.float64)
        sampler = EpsilonBallParameterSpace(center=center, epsilon=0.5)

        sample = sampler.sample()

        self.assertEqual(sample.shape, center.shape)
        self.assertEqual(sample.dtype, center.dtype)

    def test_epsilon_ball_sample_is_at_most_epsilon_from_center(self):
        center = torch.tensor(
            [[1.0, -2.0], [0.5, 3.0]],
            dtype=torch.float64,
        )
        epsilon = 0.1
        sampler = EpsilonBallParameterSpace(center=center, epsilon=epsilon)

        for _ in range(100):
            sample = sampler.sample()
            distance = torch.linalg.vector_norm(sample - center)

            self.assertLessEqual(distance.item(), epsilon + 1e-12)

    def test_epsilon_ball_does_not_change_center(self):
        center = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        original_center = center.clone()
        sampler = EpsilonBallParameterSpace(center=center, epsilon=0.2)

        center[0] = 100.0
        sample = sampler.sample()

        torch.testing.assert_close(
            sampler.center,
            original_center,
            atol=0.0,
            rtol=0.0,
        )

        self.assertLessEqual(
            torch.linalg.vector_norm(sample - original_center).item(),
            0.2 + 1e-12,
        )

    def test_epsilon_ball_is_reproducible_with_seeded_generators(self):
        center = torch.zeros((2, 2), dtype=torch.float64)

        sampler_a = EpsilonBallParameterSpace(
            center=center,
            epsilon=0.4,
            generator=torch.Generator().manual_seed(123),
        )

        sampler_b = EpsilonBallParameterSpace(
            center=center,
            epsilon=0.4,
            generator=torch.Generator().manual_seed(123),
        )

        sample_a = sampler_a.sample()
        sample_b = sampler_b.sample()

        torch.testing.assert_close(sample_a, sample_b, atol=0.0, rtol=0.0)

    def test_epsilon_must_be_positive(self):
        center = torch.zeros(2, dtype=torch.float64)

        with self.assertRaises(ValueError):
            EpsilonBallParameterSpace(center=center, epsilon=0.0)

        with self.assertRaises(ValueError):
            EpsilonBallParameterSpace(center=center, epsilon=-0.1)

    def test_transformed_radii_are_uniform(self):
        dimension = 5
        epsilon = 2.0
        n_samples = 20_000

        center = torch.zeros(dimension, dtype=torch.float64)
        sampler = EpsilonBallParameterSpace(
            center=center,
            epsilon=epsilon,
            generator=torch.Generator().manual_seed(0),
        )

        samples = torch.stack([sampler.sample() for _ in range(n_samples)])
        radii = torch.linalg.vector_norm(samples - center, dim=1)
        transformed_radii = (radii / epsilon).pow(dimension)

        expected_mean = 0.5
        expected_variance = 1.0 / 12.0

        self.assertAlmostEqual(
            transformed_radii.mean().item(),
            expected_mean,
            delta=0.01,
        )

        self.assertAlmostEqual(
            transformed_radii.var(unbiased=True).item(),
            expected_variance,
            delta=0.01,
        )

    def test_directions_are_isotropic(self):
        dimension = 5
        epsilon = 1.0
        n_samples = 20_000

        center = torch.zeros(dimension, dtype=torch.float64)
        sampler = EpsilonBallParameterSpace(
            center=center,
            epsilon=epsilon,
            generator=torch.Generator().manual_seed(1),
        )

        samples = torch.stack([sampler.sample() for _ in range(n_samples)])
        radii = torch.linalg.vector_norm(samples - center, dim=1)
        directions = (samples - center) / radii.unsqueeze(1)

        direction_mean = directions.mean(dim=0)
        empirical_second_moment = directions.T @ directions / n_samples
        expected_second_moment = torch.eye(
            dimension,
            dtype=torch.float64,
        ) / dimension

        torch.testing.assert_close(
            direction_mean,
            torch.zeros(dimension, dtype=torch.float64),
            atol=0.015,
            rtol=0.0,
        )

        torch.testing.assert_close(
            empirical_second_moment,
            expected_second_moment,
            atol=0.015,
            rtol=0.0,
        )

if __name__ == "__main__":
    unittest.main()