from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from ..core import arnoldi, lanczos


@dataclass(frozen=True)
class ComparisonResult:
    time_step: float
    hamiltonian_dimension: int
    unitary_dimension: int
    common_dimension: int
    max_principal_angle: float
    mean_principal_angle: float
    projector_distance: float
    smallest_subspace_overlap: float


def unitary_from_hamiltonian(hamiltonian: np.ndarray, time_step: float) -> np.ndarray:
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be square")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=0.0):
        raise ValueError("hamiltonian must be Hermitian")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    phases = np.exp(-1j * time_step * eigenvalues)
    return (eigenvectors * phases) @ eigenvectors.conj().T


def principal_angles(first_basis: np.ndarray, second_basis: np.ndarray) -> np.ndarray:
    first = _orthonormal_basis(first_basis)
    second = _orthonormal_basis(second_basis)
    singular_values = np.linalg.svd(first.conj().T @ second, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return np.arccos(singular_values)


def projector_distance(first_basis: np.ndarray, second_basis: np.ndarray) -> float:
    first = _orthonormal_basis(first_basis)
    second = _orthonormal_basis(second_basis)
    first_projector = first @ first.conj().T
    second_projector = second @ second.conj().T
    return float(np.linalg.norm(first_projector - second_projector, ord="fro"))


def compare_hamiltonian_and_unitary_krylov(
    hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    time_step: float,
    max_dimension: int | None = None,
    tolerance: float = 1e-12,
) -> ComparisonResult:
    hamiltonian_result = lanczos(
        hamiltonian,
        initial_state,
        max_dimension=max_dimension,
        tolerance=tolerance,
    )
    unitary = unitary_from_hamiltonian(hamiltonian, time_step)
    unitary_result = arnoldi(
        unitary,
        initial_state,
        max_dimension=max_dimension,
        tolerance=tolerance,
    )

    angles = principal_angles(hamiltonian_result.basis, unitary_result.basis)
    overlaps = np.cos(angles) ** 2
    return ComparisonResult(
        time_step=float(time_step),
        hamiltonian_dimension=hamiltonian_result.dimension,
        unitary_dimension=unitary_result.dimension,
        common_dimension=int(angles.size),
        max_principal_angle=float(np.max(angles)) if angles.size else 0.0,
        mean_principal_angle=float(np.mean(angles)) if angles.size else 0.0,
        projector_distance=projector_distance(
            hamiltonian_result.basis, unitary_result.basis
        ),
        smallest_subspace_overlap=float(np.min(overlaps)) if overlaps.size else 1.0,
    )


def run_time_step_study(
    hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    time_steps: list[float] | np.ndarray,
    output_directory: str | Path,
    max_dimension: int | None = None,
    tolerance: float = 1e-12,
) -> list[ComparisonResult]:
    results = [
        compare_hamiltonian_and_unitary_krylov(
            hamiltonian=hamiltonian,
            initial_state=initial_state,
            time_step=float(time_step),
            max_dimension=max_dimension,
            tolerance=tolerance,
        )
        for time_step in time_steps
    ]

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    fieldnames = list(rows[0]) if rows else []
    with (output / "hamiltonian_unitary_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "time_steps": [float(value) for value in time_steps],
        "max_dimension": max_dimension,
        "tolerance": tolerance,
        "matrix_dimension": int(np.asarray(hamiltonian).shape[0]),
        "note": "Hamiltonian/unitary comparison run.",
    }
    with (output / "hamiltonian_unitary_comparison_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    return results


def _orthonormal_basis(basis: np.ndarray) -> np.ndarray:
    matrix = np.asarray(basis, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("basis must be a non-empty matrix")
    gram = matrix.conj().T @ matrix
    if not np.allclose(gram, np.eye(matrix.shape[1]), atol=1e-9, rtol=0.0):
        raise ValueError("basis must be orthonormal")
    return matrix
