import numpy as np


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
IDENTITY_2 = np.eye(2, dtype=np.complex128)


def basis_state(index: int, dimension: int) -> np.ndarray:
    if not 0 <= index < dimension:
        raise ValueError("basis-state index is outside the Hilbert space")
    state = np.zeros(dimension, dtype=np.complex128)
    state[index] = 1.0
    return state


def path_hamiltonian(dimension: int, coupling: float = 1.0) -> np.ndarray:
    """Nearest-neighbour hopping Hamiltonian on a finite path."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for index in range(dimension - 1):
        matrix[index, index + 1] = coupling
        matrix[index + 1, index] = coupling
    return matrix


def two_qubit_ising(
    field_1: float = 0.7,
    field_2: float = -0.2,
    coupling: float = 1.0,
) -> np.ndarray:
    """Small dense Hamiltonian used in the example script."""
    return (
        field_1 * np.kron(SIGMA_X, IDENTITY_2)
        + field_2 * np.kron(IDENTITY_2, SIGMA_X)
        + coupling * np.kron(SIGMA_Z, SIGMA_Z)
    )
