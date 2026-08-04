"""PennyLane state-vector adapter used by the QFIM code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .core import StateFunction


@dataclass(frozen=True)
class PennyLaneStateModel:
    """State function and circuit metadata."""

    state_function: StateFunction
    weight_shape: tuple[int, ...]
    parameter_count: int
    n_qubits: int
    n_layers: int
    feature_map: str


def _require_pennylane():
    try:
        import pennylane as qml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PennyLane is required for the PennyLane QFIM adapter. "
            "Install the project dependencies with `pip install -r requirements.txt`."
        ) from exc
    return qml


def build_shared_architecture_state_model(
    inputs: Sequence[float],
    *,
    n_layers: int,
    feature_map: str = "zzfm",
    device_name: str = "default.qubit",
    feature_map_repeats: int = 1,
) -> PennyLaneStateModel:
    """Build the shared circuit with a full-state output for fixed input data."""

    if n_layers <= 0:
        raise ValueError("n_layers must be a positive integer.")
    if feature_map_repeats <= 0:
        raise ValueError("feature_map_repeats must be a positive integer.")

    x = np.asarray(inputs, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("inputs must contain at least one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError("inputs contain non-finite values.")

    qml = _require_pennylane()

    # Import only after PennyLane is confirmed to be available.  This adapter
    # reuses the current shared feature-map implementation without changing it.
    from src.models import zz_feature_map

    n_qubits = int(x.size)
    weight_shape = tuple(
        int(value)
        for value in qml.StronglyEntanglingLayers.shape(
            n_layers=n_layers,
            n_wires=n_qubits,
        )
    )
    parameter_count = int(np.prod(weight_shape))
    device = qml.device(device_name, wires=n_qubits)

    @qml.qnode(device, interface=None, diff_method=None)
    def state_circuit(flat_parameters):
        weights = qml.math.reshape(flat_parameters, weight_shape)
        wires = list(range(n_qubits))

        if feature_map == "iqp":
            qml.IQPEmbedding(x, wires=wires)
        elif feature_map == "zzfm":
            zz_feature_map(x, wires=wires, repeats=feature_map_repeats)
        elif feature_map in {"X", "Y", "Z"}:
            qml.AngleEmbedding(x, wires=wires, rotation=feature_map)
        else:
            raise ValueError(
                "feature_map must be one of: 'zzfm', 'iqp', 'X', 'Y', 'Z'."
            )

        qml.StronglyEntanglingLayers(weights, wires=wires)
        return qml.state()

    def state_function(parameters: np.ndarray) -> np.ndarray:
        theta = np.asarray(parameters, dtype=float).reshape(-1)
        if theta.size != parameter_count:
            raise ValueError(
                f"Expected {parameter_count} parameters for shape {weight_shape}, "
                f"received {theta.size}."
            )
        state = state_circuit(theta)
        return np.asarray(state, dtype=np.complex128).reshape(-1)

    return PennyLaneStateModel(
        state_function=state_function,
        weight_shape=weight_shape,
        parameter_count=parameter_count,
        n_qubits=n_qubits,
        n_layers=n_layers,
        feature_map=feature_map,
    )
