import unittest
import torch

from projects.trainability_effective_dim.core.measures.samplers import (
    UniformParameterSpace,
    EpsilonBallParameterSpace,
)


class TestSamplers(unittest.TestCase):
    def test_uniform_sample_has_expected_shape_and_dtype(self):
        """
        Verify that the uniform sampler preserves the requested tensor shape
        and data type.

        Expected result:
            - The returned tensor has shape (3, 2, 4).
            - The returned tensor uses torch.float64.
        """
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
        """
        Verify that every uniform sample lies in the requested interval.

        The sampler uses torch.uniform_, which samples from the half-open
        interval [low, high).

        Expected result:
            - Every sampled value is greater than or equal to -2.0.
            - Every sampled value is strictly less than 3.0.
        """
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
        """
        Verify that epsilon-ball samples have the same shape and data type as
        the centre tensor.

        Expected result:
            - The returned sample has shape (2, 3).
            - The returned sample has dtype torch.float64.
        """
        center = torch.zeros((2, 3), dtype=torch.float64)
        sampler = EpsilonBallParameterSpace(center=center, epsilon=0.5)

        sample = sampler.sample()

        self.assertEqual(sample.shape, center.shape)
        self.assertEqual(sample.dtype, center.dtype)

    def test_epsilon_ball_sample_is_at_most_epsilon_from_center(self):
        """
        Verify that epsilon-ball samples satisfy the Euclidean distance bound.

        Each sampled parameter tensor must lie inside, or on the boundary of,
        the ball centred at `center` with radius epsilon.

        Expected result:
            - For each of 100 samples, ||sample - center||_2 <= epsilon.
        """
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
        """
        Verify that the sampler stores an independent copy of its centre.

        The centre supplied during construction may subsequently be modified by
        training or external code. Those changes must not alter the sampling
        distribution of an already-created epsilon-ball sampler.

        Expected result:
            - sampler.center remains equal to the original centre tensor.
            - A sampled tensor remains within epsilon of the original centre,
              not the externally modified tensor.
        """
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
        """
        Verify deterministic epsilon-ball sampling with equal random seeds.

        Two independent generators initialized with the same seed should
        produce identical random directions and radii.

        Expected result:
            - The first sample produced by both samplers is exactly equal.
        """
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
        """
        Verify that invalid epsilon-ball radii are rejected.

        A radius of zero defines a degenerate ball, while a negative radius is
        mathematically invalid for this sampler.

        Expected result:
            - epsilon = 0.0 raises ValueError.
            - epsilon < 0 raises ValueError.
        """
        center = torch.zeros(2, dtype=torch.float64)

        with self.assertRaises(ValueError):
            EpsilonBallParameterSpace(center=center, epsilon=0.0)

        with self.assertRaises(ValueError):
            EpsilonBallParameterSpace(center=center, epsilon=-0.1)

    def test_transformed_radii_are_uniform(self):
        """
        Statistically verify uniform sampling over the volume of a
        five-dimensional epsilon-ball.

        For a point uniformly distributed in a d-dimensional ball, its radius
        R satisfies U = (R / epsilon)^d ~ Uniform(0, 1). Therefore, U has
        expected mean 1/2 and variance 1/12.

        Expected result:
            - The mean of transformed radii is close to 0.5.
            - The unbiased variance is close to 1/12.
            - The test detects the incorrect radius rule
              R = epsilon * Uniform(0, 1), which over-samples the centre.
        """
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
        """
        Statistically verify that epsilon-ball sampling has no preferred
        direction.

        After normalizing sample displacements, directions should be uniformly
        distributed on the unit sphere. For d dimensions, their expected mean
        is zero and their expected second-moment matrix is I / d.

        Expected result:
            - The empirical mean direction is close to the zero vector.
            - The empirical direction second-moment matrix is close to I / 5.
        """
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