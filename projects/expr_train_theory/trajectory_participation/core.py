from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryParticipationResult:
    dimension: float
    spectrum: np.ndarray
    singular_values: np.ndarray
    numerical_rank: int
    entropy: float
    gram: np.ndarray


def _as_state_matrix(states, *, atol=1e-10):
    """Return normalized state snapshots as columns of one complex matrix."""
    arr = np.asarray(states, dtype=complex)
    if arr.ndim != 2:
        raise ValueError("states must be a 2D array-like object")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("states cannot be empty")

    # The public API accepts a sequence of state vectors, so rows are snapshots.
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms <= atol):
        raise ValueError("state snapshots must have non-zero norm")

    arr = arr / norms[:, None]
    return arr.T


def trajectory_gram(states):
    """Gram matrix of normalized trajectory snapshots."""
    psi = _as_state_matrix(states)
    return psi.conj().T @ psi


def trajectory_spectrum(states, *, tol=1e-12):
    """Normalized squared singular values of the trajectory matrix."""
    psi = _as_state_matrix(states)
    singular_values = np.linalg.svd(psi, compute_uv=False)
    weights = singular_values**2
    total = float(np.sum(weights))
    if total <= tol:
        raise ValueError("trajectory has zero total weight")
    spectrum = weights / total
    spectrum = spectrum[spectrum > tol]
    return spectrum, singular_values


def trajectory_participation_dimension_from_gram(states):
    """Participation dimension from the trajectory Gram matrix."""
    gram = trajectory_gram(states)
    tr_g = float(np.trace(gram).real)
    tr_g2 = float(np.trace(gram @ gram).real)
    if tr_g2 <= 0.0:
        raise ValueError("trajectory Gram matrix has zero squared trace")
    return (tr_g * tr_g) / tr_g2


def trajectory_participation_dimension(states, *, rank_tol=1e-10):
    """Compute the equal-weight trajectory participation dimension."""
    psi = _as_state_matrix(states)
    gram = psi.conj().T @ psi
    singular_values = np.linalg.svd(psi, compute_uv=False)
    weights = singular_values**2
    spectrum = weights / np.sum(weights)

    dimension = float(1.0 / np.sum(spectrum**2))
    numerical_rank = int(np.count_nonzero(singular_values > rank_tol))

    positive = spectrum[spectrum > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))

    return TrajectoryParticipationResult(
        dimension=dimension,
        spectrum=spectrum,
        singular_values=singular_values,
        numerical_rank=numerical_rank,
        entropy=entropy,
        gram=gram,
    )
