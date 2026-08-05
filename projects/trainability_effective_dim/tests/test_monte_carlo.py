import unittest
import torch
import pennylane as qml

from projects.trainability_effective_dim.core.measures.monte_carlo import (
    sample_empirical_fishers,
)
from projects.trainability_effective_dim.core.measures.samplers import (
    UniformParameterSpace,
    EpsilonBallParameterSpace,
)


class QuantumLayer:
    def __init__(self, weights):
        self.weights = weights


class TestNet:
    def __init__(self, weights):
        self.qlayers = QuantumLayer(weights)

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, circuit_weights):
            qml.RY(inputs[0] * circuit_weights[0], wires=0)
            return qml.state()

        self._circuit = circuit

    def state_circuit(self, inputs, weights):
        return self._circuit(inputs, weights)


class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        """
        Create a one-parameter test model and a fixed set of inputs.

        The circuit is RY(x * theta)|0>. Its CFIM with respect to theta is
        exactly x^2, independently of theta. For inputs x = 1 and x = 2,
        the input-averaged empirical Fisher matrix is:

            (1^2 + 2^2) / 2 = 2.5.

        This known result allows the Monte Carlo implementation to be tested
        independently of the particular sampled parameter values.
        """
        self.initial_weights = torch.tensor(
            [0.6],
            dtype=torch.float64,
            requires_grad=True,
        )

        self.inputs = torch.tensor(
            [[1.0], [2.0]],
            dtype=torch.float64,
        )

        self.expected_fisher = torch.tensor(
            [[[2.5]]],
            dtype=torch.float64,
        )

    def test_uniform_parameter_space(self):
        """
        Verify empirical Fisher estimation with uniformly sampled parameters.

        The sampler draws eight parameter values from [0.4, 0.8). Although
        the parameter values differ, the test circuit has CFIM x^2, which is
        independent of the parameter. Therefore each input-averaged Fisher
        matrix must equal [[2.5]].

        Expected result:
            - Eight Fisher matrices are returned, with shape (8, 1, 1).
            - Every returned Fisher matrix equals [[2.5]].
            - The network weights are restored after sampling completes.
        """
        net = TestNet(self.initial_weights.clone().detach().requires_grad_(True))
        original_weights = net.qlayers.weights.detach().clone()

        sampler = UniformParameterSpace(
            low=0.4,
            high=0.8,
            shape=net.qlayers.weights.shape,
            dtype=torch.float64,
        )

        empirical_fishers = sample_empirical_fishers(
            net=net,
            parameter_space=sampler,
            inputs=self.inputs,
            n_theta=8,
        )

        self.assertEqual(empirical_fishers.shape, torch.Size([8, 1, 1]))

        expected = self.expected_fisher.repeat(8, 1, 1)

        torch.testing.assert_close(
            empirical_fishers,
            expected,
            atol=1e-10,
            rtol=1e-10,
        )

        torch.testing.assert_close(
            net.qlayers.weights,
            original_weights,
            atol=0.0,
            rtol=0.0,
        )

    def test_epsilon_ball_parameter_space(self):
        """
        Verify empirical Fisher estimation with epsilon-ball parameter samples.

        Parameters are sampled from a radius-0.1 ball around the initial
        weight. The circuit's Fisher matrix remains independent of the sampled
        parameter value, so each input-averaged result must again be [[2.5]].

        Expected result:
            - Eight Fisher matrices are returned, with shape (8, 1, 1).
            - Every returned Fisher matrix equals [[2.5]].
            - The original network weights are restored after sampling.
        """
        net = TestNet(self.initial_weights.clone().detach().requires_grad_(True))
        original_weights = net.qlayers.weights.detach().clone()

        sampler = EpsilonBallParameterSpace(
            center=net.qlayers.weights,
            epsilon=0.1,
            generator=torch.Generator().manual_seed(0),
        )

        empirical_fishers = sample_empirical_fishers(
            net=net,
            parameter_space=sampler,
            inputs=self.inputs,
            n_theta=8,
        )

        self.assertEqual(empirical_fishers.shape, torch.Size([8, 1, 1]))

        expected = self.expected_fisher.repeat(8, 1, 1)

        torch.testing.assert_close(
            empirical_fishers,
            expected,
            atol=1e-10,
            rtol=1e-10,
        )

        torch.testing.assert_close(
            net.qlayers.weights,
            original_weights,
            atol=0.0,
            rtol=0.0,
        )

    def test_one_dimensional_inputs_are_accepted(self):
        """
        Verify that a single one-dimensional input is converted into an input
        batch correctly.

        The estimator accepts a tensor of shape (1,) and internally adds a
        batch dimension. For x = 1, the circuit CFIM is exactly [[1]].

        Expected result:
            - Three Fisher matrices are returned, with shape (3, 1, 1).
            - Every Fisher matrix equals [[1.0]].
            - No explicit batch dimension is required from the caller.
        """
        net = TestNet(self.initial_weights.clone().detach().requires_grad_(True))

        sampler = UniformParameterSpace(
            low=0.4,
            high=0.8,
            shape=net.qlayers.weights.shape,
            dtype=torch.float64,
        )

        inputs = torch.tensor([1.0], dtype=torch.float64)

        empirical_fishers = sample_empirical_fishers(
            net=net,
            parameter_space=sampler,
            inputs=inputs,
            n_theta=3,
        )

        expected = torch.ones((3, 1, 1), dtype=torch.float64)

        torch.testing.assert_close(
            empirical_fishers,
            expected,
            atol=1e-10,
            rtol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()