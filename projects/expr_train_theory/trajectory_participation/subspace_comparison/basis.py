from dataclasses import dataclass

import numpy as np

from ..core import _as_state_matrix, trajectory_participation_dimension


@dataclass(frozen=True)
class SubspaceComparison:
    gram_schmidt_rank: int
    svd_rank: int
    participation_dimension: float
    rank_utilization: float
    residual_norms: np.ndarray
    basis: np.ndarray


def modified_gram_schmidt(states, *, tol=1e-10):
    """Build an orthonormal basis from trajectory snapshots in their given order."""
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    snapshots = _as_state_matrix(states).T
    basis = []
    residual_norms = []

    for state in snapshots:
        residual = state.astype(complex, copy=True)
        for vector in basis:
            residual -= np.vdot(vector, residual) * vector

        # A second pass makes the small reference routine less sensitive to
        # accumulated loss of orthogonality.
        for vector in basis:
            residual -= np.vdot(vector, residual) * vector

        norm = float(np.linalg.norm(residual))
        residual_norms.append(norm)
        if norm > tol:
            basis.append(residual / norm)

    if basis:
        basis_matrix = np.vstack(basis)
    else:
        basis_matrix = np.empty((0, snapshots.shape[1]), dtype=complex)

    return basis_matrix, np.asarray(residual_norms, dtype=float)


def compare_hard_and_soft_dimension(states, *, tol=1e-10):
    """Compare trajectory subspace rank with the soft participation dimension."""
    basis, residual_norms = modified_gram_schmidt(states, tol=tol)
    tp = trajectory_participation_dimension(states, rank_tol=tol)
    gs_rank = int(basis.shape[0])

    if gs_rank == 0:
        raise ValueError("trajectory has no non-zero directions")

    return SubspaceComparison(
        gram_schmidt_rank=gs_rank,
        svd_rank=tp.numerical_rank,
        participation_dimension=tp.dimension,
        rank_utilization=float(tp.dimension / gs_rank),
        residual_norms=residual_norms,
        basis=basis,
    )
