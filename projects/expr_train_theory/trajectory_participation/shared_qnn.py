import numpy as np

try:
    import pennylane as qml
except ImportError:  # keeps the NumPy-only parts importable
    qml = None


def _require_pennylane():
    if qml is None:
        raise ImportError("PennyLane is required for the shared QNN trajectory adapter")


def _prepare_inputs(inputs, n_qubits):
    values = np.asarray(inputs, dtype=float).reshape(-1)
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    if values.size == 0:
        raise ValueError("inputs cannot be empty")
    if values.size > n_qubits:
        values = values[-n_qubits:]
    return values


def _check_weights(weights, n_layers, n_qubits):
    values = np.asarray(weights, dtype=float)
    expected = (n_layers, n_qubits, 3)
    if values.shape != expected:
        raise ValueError(f"weights must have shape {expected}")
    return values


def _apply_encoding(inputs, wires, style):
    if style == "iqp":
        qml.IQPEmbedding(inputs, wires=wires)
    elif style == "zzfm":
        from src.models import zz_feature_map

        zz_feature_map(inputs, wires=wires)
    elif style in {"X", "Y", "Z"}:
        rotation = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}[style]
        for i, wire in enumerate(wires):
            rotation(inputs[i], wires=wire)


def _apply_reupload(inputs, wires, style):
    if style in {"X", "Y", "Z"}:
        rotation = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}[style]
        for i, wire in enumerate(wires):
            rotation(inputs[i], wires=wire)


def _apply_layer(weights, layer_index, n_qubits, feature_inputs, reup_style):
    for wire in range(n_qubits):
        qml.RZ(weights[layer_index, wire, 0], wires=wire)
        qml.RY(weights[layer_index, wire, 1], wires=wire)
        qml.RZ(weights[layer_index, wire, 2], wires=wire)

    if n_qubits > 1:
        radius = (layer_index % (n_qubits - 1)) + 1
        for wire in range(n_qubits):
            qml.CNOT(wires=[wire, (wire + radius) % n_qubits])

    if reup_style is not None:
        _apply_reupload(feature_inputs, list(range(feature_inputs.size)), reup_style)


def shared_qnn_snapshots(
    inputs,
    weights,
    *,
    n_qubits,
    fm_style="zzfm",
    reup_style=None,
    device_name="default.qubit",
):
    """Return the encoded state and one state after every shared-QNN layer."""
    _require_pennylane()

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3:
        raise ValueError("weights must have shape (layers, n_qubits, 3)")
    n_layers = weights.shape[0]
    weights = _check_weights(weights, n_layers, n_qubits)
    features = _prepare_inputs(inputs, n_qubits)
    feature_wires = list(range(features.size))

    dev = qml.device(device_name, wires=n_qubits)

    def state_after(prefix_layers):
        @qml.qnode(dev)
        def circuit():
            _apply_encoding(features, feature_wires, fm_style)
            for layer_index in range(prefix_layers):
                _apply_layer(weights, layer_index, n_qubits, features, reup_style)
            return qml.state()

        return np.asarray(circuit(), dtype=np.complex128)

    return np.asarray(
        [state_after(prefix) for prefix in range(n_layers + 1)],
        dtype=np.complex128,
    )


def z_expectation_from_state(state, wire, n_qubits):
    """Pauli-Z expectation used for a lightweight final-state check."""
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    if state.size != 2**n_qubits:
        raise ValueError("state has the wrong dimension")
    if wire < 0 or wire >= n_qubits:
        raise ValueError("wire is out of range")

    probabilities = np.abs(state) ** 2
    shift = n_qubits - wire - 1
    signs = 1.0 - 2.0 * ((np.arange(state.size) >> shift) & 1)
    return float(np.sum(signs * probabilities))
