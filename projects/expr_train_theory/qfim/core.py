"""Pure-state QFIM utilities based on NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

StateFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class QFIMDiagnostics:
    """Numerical diagnostics for a QFIM."""

    matrix: np.ndarray
    eigenvalues: np.ndarray
    numerical_rank: int
    relative_rank: float
    trace: float
    minimum_eigenvalue: float
    positive_condition_number: float
    symmetry_error: float


def _as_normalized_state(state: np.ndarray, *, normalization_tolerance: float) -> np.ndarray:
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    if vector.size == 0:
        raise ValueError("The state function returned an empty vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("The state vector contains non-finite values.")

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("The state function returned the zero vector.")
    if abs(norm - 1.0) > normalization_tolerance:
        raise ValueError(
            f"Expected a normalized state, but its norm is {norm:.12g}. "
            "Normalize the state in the model or increase normalization_tolerance deliberately."
        )
    return vector / norm


def _central_state_derivatives(
    state_function: StateFunction,
    parameters: np.ndarray,
    *,
    step: float,
    normalization_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if step <= 0.0:
        raise ValueError("The finite-difference step must be positive.")

    theta = np.asarray(parameters, dtype=float).reshape(-1)
    reference = _as_normalized_state(
        state_function(theta.copy()),
        normalization_tolerance=normalization_tolerance,
    )
    derivatives = np.empty((theta.size, reference.size), dtype=np.complex128)

    for index in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += step
        minus[index] -= step

        state_plus = _as_normalized_state(
            state_function(plus), normalization_tolerance=normalization_tolerance
        )
        state_minus = _as_normalized_state(
            state_function(minus), normalization_tolerance=normalization_tolerance
        )
        if state_plus.shape != reference.shape or state_minus.shape != reference.shape:
            raise ValueError("The state-function output dimension changed with the parameters.")

        derivatives[index] = (state_plus - state_minus) / (2.0 * step)

    return reference, derivatives


def compute_pure_state_qfim(
    state_function: StateFunction,
    parameters: np.ndarray,
    *,
    step: float = 1e-6,
    normalization_tolerance: float = 1e-8,
) -> np.ndarray:
    """Compute the pure-state QFIM using central finite differences."""

    theta = np.asarray(parameters, dtype=float).reshape(-1)
    if theta.size == 0:
        return np.zeros((0, 0), dtype=float)

    state, derivatives = _central_state_derivatives(
        state_function,
        theta,
        step=step,
        normalization_tolerance=normalization_tolerance,
    )

    derivative_gram = derivatives.conj() @ derivatives.T
    derivative_state = derivatives.conj() @ state
    projected_gram = derivative_gram - np.outer(derivative_state, derivative_state.conj())
    qfim = 4.0 * np.real(projected_gram)

    # Remove antisymmetric finite-difference noise explicitly.
    return 0.5 * (qfim + qfim.T)


def diagnose_qfim(
    matrix: np.ndarray,
    *,
    absolute_rank_tolerance: float = 1e-9,
    relative_rank_tolerance: float = 1e-7,
) -> QFIMDiagnostics:
    """Return basic numerical diagnostics for a QFIM."""

    qfim = np.asarray(matrix, dtype=float)
    if qfim.ndim != 2 or qfim.shape[0] != qfim.shape[1]:
        raise ValueError("QFIM must be a square matrix.")
    if not np.all(np.isfinite(qfim)):
        raise ValueError("QFIM contains non-finite values.")

    symmetry_error = float(np.max(np.abs(qfim - qfim.T))) if qfim.size else 0.0
    symmetric = 0.5 * (qfim + qfim.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    eigenvalues = np.sort(eigenvalues)[::-1]

    largest = float(max(0.0, eigenvalues[0])) if eigenvalues.size else 0.0
    threshold = max(absolute_rank_tolerance, relative_rank_tolerance * largest)
    positive = eigenvalues[eigenvalues > threshold]
    rank = int(positive.size)
    parameter_count = qfim.shape[0]

    condition_number = (
        float(positive[0] / positive[-1]) if positive.size >= 2 else (1.0 if positive.size == 1 else np.inf)
    )

    return QFIMDiagnostics(
        matrix=symmetric,
        eigenvalues=eigenvalues,
        numerical_rank=rank,
        relative_rank=(rank / parameter_count if parameter_count else 0.0),
        trace=float(np.trace(symmetric)),
        minimum_eigenvalue=(float(eigenvalues[-1]) if eigenvalues.size else 0.0),
        positive_condition_number=condition_number,
        symmetry_error=symmetry_error,
    )
