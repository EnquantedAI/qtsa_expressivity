import unittest
import pennylane as qml
import torch

from projects.trainability_effective_dim.core.measures.cfim import cfim_for_input


class QuantumLayer:
    def __init__(self, weights):
        self.weights = weights


class TestNet:
    def __init__(self, qnode, weights):
        self.qnode = qnode
        self.qlayers = QuantumLayer(weights)

    def state_circuit(self, inputs, weights):
        return self.qnode(inputs, weights)


class TestCFIM(unittest.TestCase):
    def test_single_ry_has_unit_fisher_information(self):
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.RY(weights[0], wires=0)
            return qml.state()

        weights = torch.tensor([0.73], dtype=torch.float64, requires_grad=True)
        inputs = torch.tensor([0.0], dtype=torch.float64)
        net = TestNet(circuit, weights)

        actual, probabilities = cfim_for_input(net, inputs)

        expected_fisher = torch.tensor([[1.0]], dtype=torch.float64)
        expected_probabilities = torch.tensor(
            [
                torch.cos(weights[0] / 2) ** 2,
                torch.sin(weights[0] / 2) ** 2,
            ],
            dtype=torch.float64,
        )

        torch.testing.assert_close(actual, expected_fisher, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(
            probabilities,
            expected_probabilities.detach(),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_shared_angle_has_rank_one_fisher_matrix(self):
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.RY(weights[0] + weights[1], wires=0)
            return qml.state()

        weights = torch.tensor(
            [0.31, 0.47],
            dtype=torch.float64,
            requires_grad=True,
        )
        inputs = torch.tensor([0.0], dtype=torch.float64)
        net = TestNet(circuit, weights)

        actual, _ = cfim_for_input(net, inputs)

        expected = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0]],
            dtype=torch.float64,
        )

        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)

    def test_independent_rotations_have_identity_fisher_matrix(self):
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            return qml.state()

        weights = torch.tensor(
            [0.42, 0.91],
            dtype=torch.float64,
            requires_grad=True,
        )
        inputs = torch.tensor([0.0], dtype=torch.float64)
        net = TestNet(circuit, weights)

        actual, _ = cfim_for_input(net, inputs)

        expected = torch.eye(2, dtype=torch.float64)

        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)

    def test_cancelling_rotations_have_zero_fisher_information(self):
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.RY(weights[0], wires=0)
            qml.RY(-weights[0], wires=0)
            return qml.state()

        weights = torch.tensor([0.63], dtype=torch.float64, requires_grad=True)
        inputs = torch.tensor([0.0], dtype=torch.float64)
        net = TestNet(circuit, weights)

        actual, _ = cfim_for_input(net, inputs)

        expected = torch.zeros((1, 1), dtype=torch.float64)

        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()