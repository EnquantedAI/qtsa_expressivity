import numpy as np

from projects.expr_train_theory.krylov.circuit_orbits.circuits import (
    embed_one_qubit,
    identity,
    layered_ansatz,
    rotation_x,
    rotation_y,
    rotation_z,
)


def angle_encoding(features, n_qubits, axis="Y"):
    """Build a small product-state angle encoding unitary."""
    values = np.asarray(features, dtype=float).reshape(-1)
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    if values.size > n_qubits:
        raise ValueError("feature vector is longer than the number of qubits")

    rotations = {"X": rotation_x, "Y": rotation_y, "Z": rotation_z}
    try:
        rotation = rotations[str(axis).upper()]
    except KeyError as exc:
        raise ValueError("axis must be X, Y or Z") from exc

    unitary = identity(n_qubits)
    for wire, value in enumerate(values):
        unitary = embed_one_qubit(rotation(float(value)), wire, n_qubits) @ unitary
    return unitary


def trajectory_snapshots(
    features,
    parameters,
    *,
    n_qubits,
    encoding_axis="Y",
    reupload_axis=None,
    entangle=True,
    initial_state=None,
):
    """Return snapshots after encoding and after each variational layer."""
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 3 or parameters.shape[1:] != (n_qubits, 3):
        raise ValueError("parameters must have shape (layers, n_qubits, 3)")

    size = 2**n_qubits
    if initial_state is None:
        state = np.zeros(size, dtype=np.complex128)
        state[0] = 1.0
    else:
        state = np.asarray(initial_state, dtype=np.complex128).reshape(-1)
        if state.size != size:
            raise ValueError("initial_state has the wrong dimension")
        norm = np.linalg.norm(state)
        if norm == 0.0 or not np.isfinite(norm):
            raise ValueError("initial_state must be finite and non-zero")
        state = state / norm

    encoding = angle_encoding(features, n_qubits, axis=encoding_axis)
    state = encoding @ state
    snapshots = [state.copy()]

    layers = layered_ansatz(parameters, n_qubits=n_qubits, entangle=entangle)
    for layer in layers:
        if reupload_axis is not None:
            reupload = angle_encoding(features, n_qubits, axis=reupload_axis)
            state = reupload @ state
        state = layer @ state
        snapshots.append(state.copy())

    return np.asarray(snapshots, dtype=np.complex128)
