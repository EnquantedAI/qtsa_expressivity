from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GramTrajectoryResult:
    dimension: float
    spectrum: np.ndarray
    numerical_rank: int
    entropy: float
    gram: np.ndarray


def gram_matrix_from_states(states, *, atol=1e-12):
    """Return the Gram matrix of normalized state snapshots.

    The public input convention is one snapshot per row.
    """
    arr = np.asarray(states, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("states must be a non-empty 2D array-like object")

    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms <= atol):
        raise ValueError("state snapshots must have non-zero norm")

    normalized = arr / norms[:, None]
    return normalized.conj() @ normalized.T


def _validate_gram(gram, *, atol=1e-10):
    matrix = np.asarray(gram, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be a non-empty square matrix")
    if not np.allclose(matrix, matrix.conj().T, atol=atol, rtol=0.0):
        raise ValueError("gram must be Hermitian")

    eigenvalues = np.linalg.eigvalsh(matrix).real
    if np.min(eigenvalues) < -atol:
        raise ValueError("gram must be positive semidefinite")
    return matrix, np.clip(eigenvalues, 0.0, None)


def spectrum_from_gram_matrix(gram, *, tol=1e-12):
    """Normalized non-zero eigenvalue spectrum of a trajectory Gram matrix."""
    matrix, eigenvalues = _validate_gram(gram)
    total = float(np.sum(eigenvalues))
    if total <= tol:
        raise ValueError("gram matrix has zero trace")

    spectrum = eigenvalues / total
    spectrum = spectrum[spectrum > tol]
    return spectrum[::-1]


def participation_dimension_from_gram_matrix(gram, *, tol=1e-12):
    """Compute d_TP directly from a precomputed Gram matrix."""
    matrix, _ = _validate_gram(gram)
    trace = float(np.trace(matrix).real)
    trace_square = float(np.trace(matrix @ matrix).real)
    if trace <= tol or trace_square <= tol:
        raise ValueError("gram matrix has zero trajectory weight")
    return (trace * trace) / trace_square


def analyse_trajectory_from_gram(gram, *, rank_tol=1e-10):
    """Return the main trajectory diagnostics using only the Gram matrix."""
    matrix, eigenvalues = _validate_gram(gram)
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        raise ValueError("gram matrix has zero trace")

    spectrum_full = eigenvalues / total
    positive = spectrum_full[spectrum_full > 0.0]
    dimension = float(1.0 / np.sum(spectrum_full**2))
    numerical_rank = int(np.count_nonzero(eigenvalues > rank_tol))
    entropy = float(-np.sum(positive * np.log(positive)))

    return GramTrajectoryResult(
        dimension=dimension,
        spectrum=np.sort(spectrum_full)[::-1],
        numerical_rank=numerical_rank,
        entropy=entropy,
        gram=matrix,
    )
