from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LanczosResult:
    basis: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    tridiagonal: np.ndarray
    residual_norm: float

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]


@dataclass(frozen=True)
class ArnoldiResult:
    basis: np.ndarray
    hessenberg: np.ndarray
    residual_norm: float

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]


def _as_square_matrix(operator: np.ndarray) -> np.ndarray:
    matrix = np.asarray(operator, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("operator contains non-finite entries")
    return matrix


def _normalise(vector: np.ndarray, expected_size: int) -> np.ndarray:
    state = np.asarray(vector, dtype=np.complex128).reshape(-1)
    if state.size != expected_size:
        raise ValueError("starting vector has the wrong dimension")
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("starting vector must have a finite, non-zero norm")
    return state / norm


def lanczos(
    operator: np.ndarray,
    start_vector: np.ndarray,
    max_dimension: int | None = None,
    tolerance: float = 1e-12,
    reorthogonalize: bool = True,
) -> LanczosResult:
    """Build a state Krylov basis for a Hermitian operator."""
    matrix = _as_square_matrix(operator)
    size = matrix.shape[0]
    if not np.allclose(matrix, matrix.conj().T, atol=10 * tolerance, rtol=0.0):
        raise ValueError("Lanczos requires a Hermitian operator")

    if max_dimension is None:
        max_dimension = size
    if not 1 <= max_dimension <= size:
        raise ValueError("max_dimension must be between 1 and the matrix size")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    basis_vectors: list[np.ndarray] = []
    alpha: list[float] = []
    beta: list[float] = []

    current = _normalise(start_vector, size)
    previous = np.zeros(size, dtype=np.complex128)
    previous_beta = 0.0
    last_residual = 0.0

    for index in range(max_dimension):
        basis_vectors.append(current)
        work = matrix @ current
        diagonal = np.vdot(current, work)
        if abs(diagonal.imag) > 100 * tolerance:
            raise RuntimeError("Lanczos diagonal coefficient is unexpectedly complex")
        alpha.append(float(diagonal.real))

        work = work - diagonal.real * current
        if index > 0:
            work = work - previous_beta * previous

        if reorthogonalize:
            # A second pass keeps the basis stable for small dense problems.
            for _ in range(2):
                for vector in basis_vectors:
                    work = work - np.vdot(vector, work) * vector

        residual = float(np.linalg.norm(work))
        last_residual = residual
        if residual <= tolerance or index + 1 == max_dimension:
            break

        beta.append(residual)
        previous, current = current, work / residual
        previous_beta = residual

    basis = np.column_stack(basis_vectors)
    dimension = basis.shape[1]
    tridiagonal = np.diag(np.asarray(alpha, dtype=float))
    if dimension > 1:
        off_diagonal = np.asarray(beta[: dimension - 1], dtype=float)
        tridiagonal += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)

    return LanczosResult(
        basis=basis,
        alpha=np.asarray(alpha, dtype=float),
        beta=np.asarray(beta[: max(0, dimension - 1)], dtype=float),
        tridiagonal=tridiagonal,
        residual_norm=last_residual,
    )


def arnoldi(
    operator: np.ndarray,
    start_vector: np.ndarray,
    max_dimension: int | None = None,
    tolerance: float = 1e-12,
) -> ArnoldiResult:
    """Build a Krylov basis for a general square operator."""
    matrix = _as_square_matrix(operator)
    size = matrix.shape[0]
    if max_dimension is None:
        max_dimension = size
    if not 1 <= max_dimension <= size:
        raise ValueError("max_dimension must be between 1 and the matrix size")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    basis = np.zeros((size, max_dimension), dtype=np.complex128)
    hessenberg = np.zeros((max_dimension, max_dimension), dtype=np.complex128)
    basis[:, 0] = _normalise(start_vector, size)
    dimension = 1
    last_residual = 0.0

    for column in range(max_dimension):
        work = matrix @ basis[:, column]
        for row in range(column + 1):
            hessenberg[row, column] = np.vdot(basis[:, row], work)
            work -= hessenberg[row, column] * basis[:, row]

        # Repeat the projection once to reduce loss of orthogonality.
        for row in range(column + 1):
            correction = np.vdot(basis[:, row], work)
            hessenberg[row, column] += correction
            work -= correction * basis[:, row]

        residual = float(np.linalg.norm(work))
        last_residual = residual
        if column + 1 == max_dimension or residual <= tolerance:
            dimension = column + 1
            break

        hessenberg[column + 1, column] = residual
        basis[:, column + 1] = work / residual
        dimension = column + 2

    return ArnoldiResult(
        basis=basis[:, :dimension],
        hessenberg=hessenberg[:dimension, :dimension],
        residual_norm=last_residual,
    )
