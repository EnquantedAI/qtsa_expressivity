import numpy as np

from .core import LanczosResult


def _hermitian_evolution(matrix: np.ndarray, time: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    phases = np.exp(-1j * time * eigenvalues)
    return (eigenvectors * phases) @ eigenvectors.conj().T


def exact_state(
    hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    time: float,
) -> np.ndarray:
    """Evolve a state exactly under a small dense Hermitian Hamiltonian."""
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    state = np.asarray(initial_state, dtype=np.complex128).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be square")
    if state.size != matrix.shape[0]:
        raise ValueError("state and Hamiltonian dimensions do not match")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=0.0):
        raise ValueError("hamiltonian must be Hermitian")
    norm = np.linalg.norm(state)
    if norm == 0.0:
        raise ValueError("initial_state must be non-zero")
    state = state / norm
    return _hermitian_evolution(matrix, time) @ state


def projected_state(result: LanczosResult, time: float) -> np.ndarray:
    """Evolve the first Krylov vector with the projected Hamiltonian."""
    first = np.zeros(result.dimension, dtype=np.complex128)
    first[0] = 1.0
    coefficients = _hermitian_evolution(result.tridiagonal, time) @ first
    return result.basis @ coefficients
