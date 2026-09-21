from collections.abc import Mapping

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


# Explicit name for the independently reconstructed reference path.  Keep the
# original public name for backwards compatibility with the existing studies.
reference_shared_qnn_snapshots = shared_qnn_snapshots


def _extract_depth_snapshots(snapshot_results, n_layers, n_qubits):
    """Return ``depth_0`` ... ``depth_L`` snapshots as a state matrix.

    ``qml.snapshots`` may include additional entries such as the final execution
    result.  The trajectory adapter deliberately selects only the named depth
    snapshots emitted by ``src.models.gqnn(add_snaps=True)``.
    """
    if not isinstance(snapshot_results, Mapping):
        raise TypeError("snapshot_results must be a mapping")
    if n_layers < 0:
        raise ValueError("n_layers cannot be negative")
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")

    expected_dim = 2**n_qubits
    states = []
    for depth in range(n_layers + 1):
        key = f"depth_{depth}"
        if key not in snapshot_results:
            raise KeyError(f"missing shared-QNN snapshot {key!r}")
        state = np.asarray(snapshot_results[key], dtype=np.complex128).reshape(-1)
        if state.size != expected_dim:
            raise ValueError(
                f"snapshot {key!r} has dimension {state.size}, expected {expected_dim}"
            )
        states.append(state)

    return np.asarray(states, dtype=np.complex128)


def native_shared_qnn_snapshots(
    inputs,
    weights,
    *,
    n_qubits,
    fm_style="zzfm",
    reup_style=None,
    device_name="default.qubit",
    interface="autograd",
    diff_method=None,
):
    """Collect trajectory states directly from the shared PennyLane QNN.

    This is the integration path for experiments that should use the team's
    canonical ``src.models.gqnn`` circuit.  The shared model is instantiated
    with ``add_snaps=True`` and its ``depth_0`` ... ``depth_L`` states are
    collected with ``qml.snapshots``.  No circuit layers are reconstructed in
    this function.
    """
    _require_pennylane()

    values = np.asarray(weights, dtype=float)
    if values.ndim != 3:
        raise ValueError("weights must have shape (layers, n_qubits, 3)")
    n_layers = values.shape[0]
    values = _check_weights(values, n_layers, n_qubits)

    # Keep the original input shape here: src.models.gqnn owns the input
    # trimming semantics, so this adapter follows the shared implementation
    # rather than duplicating them.
    feature_values = np.asarray(inputs, dtype=float).reshape(-1)
    if feature_values.size == 0:
        raise ValueError("inputs cannot be empty")

    from src.models import gqnn

    dev = qml.device(device_name, wires=n_qubits)
    circuit = gqnn(
        n_layers,
        n_qubits,
        dev,
        interface=interface,
        diff_method=diff_method,
        fm_style=fm_style,
        reup_style=reup_style,
        meas=[0],
        add_snaps=True,
    )
    snapshot_results = qml.snapshots(circuit)(feature_values, values)
    return _extract_depth_snapshots(snapshot_results, n_layers, n_qubits)


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
