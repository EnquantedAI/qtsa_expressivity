import torch
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass

from projects.expr_train_theory.qfim.core import StateFunction

@dataclass(frozen=True)
class PennyLaneStateModel:
    """State function and circuit metadata."""
    state_function: StateFunction
    weight_shape: tuple[int, ...]
    parameter_count: int
    n_qubits: int
    n_layers: int
    feature_map: str


class GQNN:
    def __init__(
        self,
        n_layers,
        n_qubits,
        quantum_device,
        interface="torch",
        diff_method="torch",
        fm_style="zzfm",
        meas=[0],
    ):
        if n_layers <= 0:
            raise ValueError("n_layers must be a positive integer.")

        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.quantum_device = quantum_device
        self.interface = interface
        self.diff_method = diff_method
        self.fm_style = fm_style
        self.meas = meas

        self.weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=n_layers,
            n_wires=n_qubits,
        )
        self.weight_shapes_torch_interface = {"weights": self.weight_shape}
        self.parameter_count = int(np.prod(self.weight_shape))

        def ansatz(inputs, weights):
            wires = list(range(self.n_qubits))

            if self.fm_style == "iqp":
                qml.IQPEmbedding(inputs, wires=wires)
            elif self.fm_style == "zzfm":
                self.zz_feature_map(inputs, wires=wires)
            elif self.fm_style in {"X", "Y", "Z"}:
                qml.AngleEmbedding(
                    inputs,
                    wires=wires,
                    rotation=self.fm_style,
                )
            else:
                raise ValueError("fm_style must be 'zzfm', 'iqp', 'X', 'Y', or 'Z'.")

            qml.StronglyEntanglingLayers(weights, wires=wires)

        @qml.qnode(quantum_device, interface=interface, diff_method=diff_method)
        def circuit(inputs, weights):
            ansatz(inputs, weights)
            return [qml.expval(qml.PauliZ(m)) for m in self.meas]

        @qml.qnode(quantum_device, interface="torch", diff_method=diff_method)
        def state_circuit(inputs, weights):
            ansatz(inputs, weights)
            return qml.state()


        self.circuit = circuit
        self.state_circuit = state_circuit
        self.qlayers = qml.qnn.TorchLayer(circuit, self.weight_shapes)


    ### Creates a ZZ feature map (Qiskit style)
    @staticmethod
    def zz_feature_map(x, wires, repeats=1):
        """
        Implements a second-order Havlíček ZZ feature map in PennyLane.

        Parameters:
        -----------
        x : array-like
            The input classical data vector. Length must match the number of wires.
        wires : list or iterable
            The quantum wires/qubits to apply the feature map on.
        repeats : int
            Number of times to repeat the feature map layer (depth).
        """
        num_qubits = len(x)  # There could be more wires than data

        for _ in range(repeats):
            # 1. Uniform Superposition Layer
            for i in range(num_qubits):
                qml.Hadamard(wires=wires[i])

            # 2. First-Order Terms: Single-qubit Pauli-Z rotations
            for i in range(num_qubits):
                # PennyLane Rz(theta) applies exp(-i * theta * Z / 2)
                # To get exp(i * x_i * Z), we pass theta = -2 * x_i
                qml.RZ(-2.0 * x[i], wires=wires[i])

            # 3. Second-Order Terms: Pairwise ZZ interactions
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    # Calculate the Havlíček coupling coefficient
                    phi_ij = (np.pi - x[i]) * (np.pi - x[j])

                    # Implement exp(i * phi_ij * Z_i * Z_j) using CNOT-Rz-CNOT
                    qml.CNOT(wires=[wires[i], wires[j]])
                    qml.RZ(-2 * phi_ij, wires=wires[j])
                    qml.CNOT(wires=[wires[i], wires[j]])


    ### Returns the shape of the model weights
    @staticmethod
    def gqnn_shape(n_layers, n_qubits):
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        return shape

    ### Draw this circuit beautifully as in Qiskit
    #   Lots of styles apply, e.g. 'black_white', 'black_white_dark', 'sketch',
    #     'pennylane', 'pennylane_sketch', 'sketch_dark', 'solarized_light', 'solarized_dark',
    #     'default', we can even use 'rcParams' to redefine all attributes
    #   level = None, 'user', 'top', 'device', 'gradient', 0, 1, ...
    def draw_circuit(self, fontsize=20, style='pennylane',
                     scale=None, title=None, decimals=2, level='user'):
        def _draw_circuit(*args, **kwargs):
            nonlocal fontsize, style, scale, title, level
            qml.drawer.use_style(style)
            fig, ax = qml.draw_mpl(self.circuit, decimals=decimals, level=level)(*args, **kwargs)
            if scale is not None:
                dpi = fig.get_dpi()
                fig.set_dpi(dpi * scale)
            if title is not None:
                fig.suptitle(title, fontsize=fontsize)
            plt.show()

        return _draw_circuit

    def state_function(self, inputs, flat_parameters):
        inputs = torch.as_tensor(
            inputs,
            dtype=torch.float64,
            device=self.qlayers.weights.device,
        )

        theta = torch.as_tensor(
            flat_parameters,
            dtype=self.qlayers.weights.dtype,
            device=self.qlayers.weights.device,
        ).reshape(-1)

        if theta.numel() != self.parameter_count:
            raise ValueError(
                f"Expected {self.parameter_count} parameters, "
                f"received {theta.numel()}."
            )

        weights = theta.reshape(self.weight_shape)
        return self.state_circuit(inputs, weights)