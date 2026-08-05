from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from ..comparison.study import principal_angles, projector_distance
from ..core import arnoldi


@dataclass(frozen=True)
class CircuitOrbitResult:
    layer_count: int
    hilbert_dimension: int
    layer_orbit_dimension: int
    repeated_cycle_dimension: int
    common_dimension: int
    projector_distance: float
    max_principal_angle: float
    smallest_subspace_overlap: float
    layer_gram_condition: float


def cumulative_states(
    layers: list[np.ndarray],
    initial_state: np.ndarray,
    include_initial: bool = True,
) -> np.ndarray:
    if not layers:
        raise ValueError("at least one layer is required")
    first = np.asarray(layers[0], dtype=np.complex128)
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("layers must be square matrices")
    size = first.shape[0]
    state = _normalise(initial_state, size)
    states = [state.copy()] if include_initial else []

    for layer in layers:
        matrix = np.asarray(layer, dtype=np.complex128)
        if matrix.shape != (size, size):
            raise ValueError("all layers must have the same shape")
        if not np.allclose(matrix.conj().T @ matrix, np.eye(size), atol=1e-10, rtol=0.0):
            raise ValueError("each layer must be unitary")
        state = matrix @ state
        states.append(state.copy())
    return np.column_stack(states)


def orthonormal_span(vectors: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("vectors must be a non-empty matrix")
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        raise ValueError("could not build a span")
    threshold = tolerance * max(matrix.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank == 0:
        raise ValueError("vectors have zero numerical rank")
    return left[:, :rank]


def cycle_unitary(layers: list[np.ndarray]) -> np.ndarray:
    first = np.asarray(layers[0], dtype=np.complex128)
    result = np.eye(first.shape[0], dtype=np.complex128)
    for layer in layers:
        result = np.asarray(layer, dtype=np.complex128) @ result
    return result


def analyse_layer_orbit(
    layers: list[np.ndarray],
    initial_state: np.ndarray,
    tolerance: float = 1e-12,
) -> CircuitOrbitResult:
    states = cumulative_states(layers, initial_state, include_initial=True)
    layer_basis = orthonormal_span(states, tolerance=tolerance)
    cycle = cycle_unitary(layers)
    repeated = arnoldi(
        cycle,
        initial_state,
        max_dimension=cycle.shape[0],
        tolerance=tolerance,
    )
    angles = principal_angles(layer_basis, repeated.basis)
    overlaps = np.cos(angles) ** 2

    gram = states.conj().T @ states
    eigenvalues = np.linalg.eigvalsh(gram)
    positive = eigenvalues[eigenvalues > tolerance]
    condition = float(positive.max() / positive.min()) if positive.size else float("inf")

    return CircuitOrbitResult(
        layer_count=len(layers),
        hilbert_dimension=cycle.shape[0],
        layer_orbit_dimension=layer_basis.shape[1],
        repeated_cycle_dimension=repeated.dimension,
        common_dimension=int(angles.size),
        projector_distance=projector_distance(layer_basis, repeated.basis),
        max_principal_angle=float(np.max(angles)) if angles.size else 0.0,
        smallest_subspace_overlap=float(np.min(overlaps)) if overlaps.size else 1.0,
        layer_gram_condition=condition,
    )


def run_parameter_study(
    parameter_sets: list[np.ndarray],
    n_qubits: int,
    initial_state: np.ndarray,
    layer_builder,
    output_directory: str | Path,
    tolerance: float = 1e-12,
) -> list[CircuitOrbitResult]:
    results = [
        analyse_layer_orbit(
            layer_builder(parameters, n_qubits),
            initial_state,
            tolerance=tolerance,
        )
        for parameters in parameter_sets
    ]

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with (output / "circuit_orbit_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "circuit_orbit_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_qubits": n_qubits,
                "runs": len(results),
                "tolerance": tolerance,
                "note": "Circuit-orbit diagnostic.",
            },
            handle,
            indent=2,
        )
    return results


def _normalise(vector: np.ndarray, size: int) -> np.ndarray:
    state = np.asarray(vector, dtype=np.complex128).reshape(-1)
    if state.size != size:
        raise ValueError("initial state has the wrong dimension")
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("initial state must be finite and non-zero")
    return state / norm
