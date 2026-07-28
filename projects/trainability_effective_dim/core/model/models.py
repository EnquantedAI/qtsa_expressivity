import torch
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Union, Callable, Any

from projects.expr_train_theory.qfim.core import StateFunction


@dataclass(frozen=True)
class PennyLaneStateModel:
    """
    State function and circuit metadata.
    """
    state_function: StateFunction
    weight_shape: tuple[int, ...]
    parameter_count: int
    n_qubits: int
    n_layers: int
    feature_map: str


class GQNN:
    """
    Generalized Quantum Neural Network class encapsulating a PennyLane circuit
    compatible with PyTorch and state vector simulation.
    """

    def __init__(
            self,
            n_layers: int,
            n_qubits: int,
            quantum_device: qml.devices,
            interface: str = "torch",
            diff_method: str = "torch",
            fm_style: str = "zzfm",
            meas: List[int] = [0],
    ) -> None:
        """
        Initialize the Generalized Quantum Neural Network.

        Args:
            n_layers (int): Number of layers for the strongly entangling ansatz.
            n_qubits (int): Number of qubits in the circuit.
            quantum_device (qml.Device): PennyLane device used for quantum execution.
            interface (str, optional): Autodifferentiation interface. Defaults to "torch".
            diff_method (str, optional): Differentiation method. Defaults to "torch".
            fm_style (str, optional): Feature map style ('zzfm', 'iqp', 'X', 'Y', 'Z'). Defaults to "zzfm".
            meas (List[int], optional): List of qubit indices to measure. Defaults to [0].
        """
        if n_layers <= 0:
            raise ValueError("n_layers must be a positive integer.")
        if n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer.")
        if not isinstance(meas, (list, tuple)) or not all(isinstance(m, int) for m in meas):
            raise TypeError("meas must be a list or tuple of integers.")

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

        def ansatz(inputs: torch.Tensor, weights: torch.Tensor) -> None:
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
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> List[torch.Tensor]:
            ansatz(inputs, weights)
            return [qml.expval(qml.PauliZ(m)) for m in self.meas]

        @qml.qnode(quantum_device, interface="torch", diff_method=diff_method)
        def state_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            ansatz(inputs, weights)
            return qml.state()

        self.circuit = circuit
        self.state_circuit = state_circuit
        self.PennyLaneStateModel =  PennyLaneStateModel(
            state_function=self.state_function,
            weight_shape=self.weight_shape,
            parameter_count=self.parameter_count,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            feature_map=self.fm_style,
        )
        self.qlayers = qml.qnn.TorchLayer(circuit, self.weight_shapes_torch_interface)

    @staticmethod
    def zz_feature_map(
            x: Union[torch.Tensor, np.ndarray, List[float]],
            wires: List[int],
            repeats: int = 1
    ) -> None:
        """
        Implement a second-order Havlicek ZZ feature map in PennyLane.

        Args:
            x (Union[torch.Tensor, np.ndarray, List[float]]): The input classical data vector.
                Length must match the number of wires.
            wires (List[int]): The quantum wires to apply the feature map on.
            repeats (int, optional): Number of times to repeat the feature map layer. Defaults to 1.
        """
        if len(x) != len(wires):
            raise ValueError(f"Length of inputs ({len(x)}) must match number of wires ({len(wires)}).")
        if repeats <= 0:
            raise ValueError("repeats must be a positive integer.")

        num_qubits = len(x)

        for _ in range(repeats):
            for i in range(num_qubits):
                qml.Hadamard(wires=wires[i])

            for i in range(num_qubits):
                qml.RZ(-2.0 * x[i], wires=wires[i])

            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    phi_ij = (np.pi - x[i]) * (np.pi - x[j])

                    qml.CNOT(wires=[wires[i], wires[j]])
                    qml.RZ(-2 * phi_ij, wires=wires[j])
                    qml.CNOT(wires=[wires[i], wires[j]])

    @staticmethod
    def gqnn_shape(n_layers: int, n_qubits: int) -> Tuple[int, int, int]:
        """
        Return the shape of the model weights for strongly entangling layers.

        Args:
            n_layers (int): Number of circuit layers.
            n_qubits (int): Number of qubits.

        Returns:
            Tuple[int, int, int]: The shape tuple for the weight tensor.
        """
        if n_layers <= 0 or n_qubits <= 0:
            raise ValueError("n_layers and n_qubits must be positive integers.")

        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        return shape

    def draw_circuit(
            self,
            fontsize: int = 20,
            style: str = 'pennylane',
            scale: Union[float, None] = None,
            title: Union[str, None] = None,
            decimals: int = 2,
            level: Union[str, int] = 'user'
    ) -> Callable:
        """
        Create a callable function that renders the circuit architecture using matplotlib.

        Args:
            fontsize (int, optional): Font size for the plot title. Defaults to 20.
            style (str, optional): PennyLane drawer style. Defaults to 'pennylane'.
            scale (float, optional): Scaling factor for the figure resolution. Defaults to None.
            title (str, optional): Plot title. Defaults to None.
            decimals (int, optional): Number of decimal places for parameter display. Defaults to 2.
            level (Union[str, int], optional): PennyLane drawer level. Defaults to 'user'.

        Returns:
            Callable: A function executing the drawing process.
        """

        def _draw_circuit(*args: Any, **kwargs: Any) -> None:
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

    def state_function(
            self,
            inputs: Union[torch.Tensor, np.ndarray, List[float]],
    ) -> PennyLaneStateModel:
        """
        Create a NumPy-compatible state function for QFIM evaluation.
        """
        if isinstance(inputs, torch.Tensor):
            fixed_inputs = inputs.detach().cpu().numpy()
        else:
            fixed_inputs = np.asarray(inputs, dtype=float)

        fixed_inputs = fixed_inputs.reshape(-1)

        if fixed_inputs.size != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} input features, received {fixed_inputs.size}.")
        if not np.all(np.isfinite(fixed_inputs)):
            raise ValueError("inputs must contain only finite values.")

        def evaluate_state(flat_parameters: np.ndarray) -> np.ndarray:
            theta = np.asarray(flat_parameters, dtype=float).reshape(-1)

            if theta.size != self.parameter_count:
                raise ValueError(
                    f"Expected {self.parameter_count} parameters, "
                    f"received {theta.size}."
                )

            weights = theta.reshape(self.weight_shape)
            state = self.state_circuit(fixed_inputs, weights)

            return np.asarray(state, dtype=np.complex128).reshape(-1)

        return PennyLaneStateModel(
            state_function=evaluate_state,
            weight_shape=self.weight_shape,
            parameter_count=self.parameter_count,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            feature_map=self.fm_style,
        )