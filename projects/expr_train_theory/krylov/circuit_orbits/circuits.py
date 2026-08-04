import numpy as np


def identity(n_qubits: int) -> np.ndarray:
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    return np.eye(2**n_qubits, dtype=np.complex128)


def pauli_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def hadamard() -> np.ndarray:
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)


def rotation_x(angle: float) -> np.ndarray:
    half = float(angle) / 2.0
    return np.cos(half) * np.eye(2) - 1j * np.sin(half) * pauli_x()


def rotation_y(angle: float) -> np.ndarray:
    half = float(angle) / 2.0
    return np.array(
        [[np.cos(half), -np.sin(half)], [np.sin(half), np.cos(half)]],
        dtype=np.complex128,
    )


def rotation_z(angle: float) -> np.ndarray:
    half = float(angle) / 2.0
    return np.diag([np.exp(-1j * half), np.exp(1j * half)]).astype(np.complex128)


def cnot(control: int, target: int, n_qubits: int) -> np.ndarray:
    if n_qubits < 2:
        raise ValueError("CNOT requires at least two qubits")
    if control == target or not (0 <= control < n_qubits) or not (0 <= target < n_qubits):
        raise ValueError("invalid control or target")

    size = 2**n_qubits
    matrix = np.zeros((size, size), dtype=np.complex128)
    for column in range(size):
        bits = _index_to_bits(column, n_qubits)
        output = bits.copy()
        if bits[control] == 1:
            output[target] ^= 1
        row = _bits_to_index(output)
        matrix[row, column] = 1.0
    return matrix


def embed_one_qubit(gate: np.ndarray, wire: int, n_qubits: int) -> np.ndarray:
    gate = np.asarray(gate, dtype=np.complex128)
    if gate.shape != (2, 2):
        raise ValueError("gate must be a 2x2 matrix")
    if not 0 <= wire < n_qubits:
        raise ValueError("wire out of range")

    factors = [gate if index == wire else np.eye(2) for index in range(n_qubits)]
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return np.asarray(result, dtype=np.complex128)


def layered_ansatz(
    parameters: np.ndarray,
    n_qubits: int,
    entangle: bool = True,
) -> list[np.ndarray]:
    """Return one dense unitary per layer for a small RX-RY-RZ ansatz."""
    values = np.asarray(parameters, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (n_qubits, 3):
        raise ValueError("parameters must have shape (layers, n_qubits, 3)")

    layers: list[np.ndarray] = []
    for layer_parameters in values:
        layer = identity(n_qubits)
        for wire, (rx, ry, rz) in enumerate(layer_parameters):
            rotation = rotation_z(rz) @ rotation_y(ry) @ rotation_x(rx)
            layer = embed_one_qubit(rotation, wire, n_qubits) @ layer
        if entangle and n_qubits > 1:
            for control in range(n_qubits - 1):
                layer = cnot(control, control + 1, n_qubits) @ layer
        layers.append(layer)
    return layers


def _index_to_bits(index: int, n_qubits: int) -> list[int]:
    return [int(bit) for bit in format(index, f"0{n_qubits}b")]


def _bits_to_index(bits: list[int]) -> int:
    return int("".join(str(bit) for bit in bits), 2)
