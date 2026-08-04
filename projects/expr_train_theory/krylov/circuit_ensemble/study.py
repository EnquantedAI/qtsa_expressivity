from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ..circuit_orbits.circuits import (
    cnot,
    embed_one_qubit,
    identity,
    rotation_x,
    rotation_y,
    rotation_z,
)
from ..circuit_orbits.study import analyse_layer_orbit, cumulative_states


_ALLOWED_TOPOLOGIES = {"none", "linear", "ring", "full"}


@dataclass(frozen=True)
class EnsembleConfig:
    n_qubits: int
    depths: tuple[int, ...]
    topologies: tuple[str, ...]
    samples: int = 20
    seed: int = 2026
    parameter_low: float = -np.pi
    parameter_high: float = np.pi
    tolerance: float = 1e-12

    def validate(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if not self.depths or any(depth < 1 for depth in self.depths):
            raise ValueError("depths must contain positive integers")
        if not self.topologies:
            raise ValueError("at least one topology is required")
        unknown = set(self.topologies) - _ALLOWED_TOPOLOGIES
        if unknown:
            raise ValueError(f"unknown topology: {sorted(unknown)}")
        if self.samples < 1:
            raise ValueError("samples must be positive")
        if not self.parameter_low < self.parameter_high:
            raise ValueError("parameter_low must be smaller than parameter_high")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


@dataclass(frozen=True)
class EnsembleRecord:
    n_qubits: int
    depth: int
    topology: str
    sample: int
    seed: int
    hilbert_dimension: int
    layer_orbit_dimension: int
    repeated_cycle_dimension: int
    layer_orbit_fraction: float
    repeated_cycle_fraction: float
    projector_distance: float
    max_principal_angle: float
    smallest_subspace_overlap: float
    layer_gram_condition: float
    final_mean_single_qubit_entropy: float
    max_mean_single_qubit_entropy: float


def build_layers(
    parameters: np.ndarray,
    n_qubits: int,
    topology: str,
) -> list[np.ndarray]:
    values = np.asarray(parameters, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (n_qubits, 3):
        raise ValueError("parameters must have shape (depth, n_qubits, 3)")
    if topology not in _ALLOWED_TOPOLOGIES:
        raise ValueError(f"unknown topology: {topology}")

    pairs = _entangling_pairs(n_qubits, topology)
    layers: list[np.ndarray] = []
    for layer_parameters in values:
        layer = identity(n_qubits)
        for wire, (rx, ry, rz) in enumerate(layer_parameters):
            gate = rotation_z(rz) @ rotation_y(ry) @ rotation_x(rx)
            layer = embed_one_qubit(gate, wire, n_qubits) @ layer
        for control, target in pairs:
            layer = cnot(control, target, n_qubits) @ layer
        layers.append(layer)
    return layers


def run_ensemble_study(
    config: EnsembleConfig,
    output_directory: str | Path,
    initial_state: np.ndarray | None = None,
) -> list[EnsembleRecord]:
    config.validate()
    dimension = 2**config.n_qubits
    state = _default_state(dimension) if initial_state is None else _normalise(initial_state, dimension)
    rng = np.random.default_rng(config.seed)
    records: list[EnsembleRecord] = []

    for topology in config.topologies:
        for depth in config.depths:
            for sample in range(config.samples):
                sample_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
                sample_rng = np.random.default_rng(sample_seed)
                parameters = sample_rng.uniform(
                    config.parameter_low,
                    config.parameter_high,
                    size=(depth, config.n_qubits, 3),
                )
                layers = build_layers(parameters, config.n_qubits, topology)
                result = analyse_layer_orbit(
                    layers,
                    state,
                    tolerance=config.tolerance,
                )
                visited_states = cumulative_states(layers, state, include_initial=True)
                entropies = np.array([
                    mean_single_qubit_entropy(visited_states[:, index], config.n_qubits)
                    for index in range(visited_states.shape[1])
                ])
                records.append(
                    EnsembleRecord(
                        n_qubits=config.n_qubits,
                        depth=depth,
                        topology=topology,
                        sample=sample,
                        seed=sample_seed,
                        hilbert_dimension=result.hilbert_dimension,
                        layer_orbit_dimension=result.layer_orbit_dimension,
                        repeated_cycle_dimension=result.repeated_cycle_dimension,
                        layer_orbit_fraction=result.layer_orbit_dimension / result.hilbert_dimension,
                        repeated_cycle_fraction=result.repeated_cycle_dimension / result.hilbert_dimension,
                        projector_distance=result.projector_distance,
                        max_principal_angle=result.max_principal_angle,
                        smallest_subspace_overlap=result.smallest_subspace_overlap,
                        layer_gram_condition=result.layer_gram_condition,
                        final_mean_single_qubit_entropy=float(entropies[-1]),
                        max_mean_single_qubit_entropy=float(entropies.max()),
                    )
                )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "circuit_ensemble_raw.csv", [asdict(record) for record in records])
    summary = summarise_records(records)
    _write_csv(output / "circuit_ensemble_summary.csv", summary)
    with (output / "circuit_ensemble_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **asdict(config),
                "depths": list(config.depths),
                "topologies": list(config.topologies),
                "initial_state": "computational |0...0>" if initial_state is None else "user supplied",
                "note": "Orbit-space ensemble check.",
            },
            handle,
            indent=2,
        )
    return records


def summarise_records(records: Iterable[EnsembleRecord]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[int, str], list[EnsembleRecord]] = {}
    for record in records:
        grouped.setdefault((record.depth, record.topology), []).append(record)

    rows: list[dict[str, float | int | str]] = []
    for (depth, topology), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        hilbert_dimension = group[0].hilbert_dimension
        layer_dimensions = np.array([entry.layer_orbit_dimension for entry in group], dtype=float)
        cycle_dimensions = np.array([entry.repeated_cycle_dimension for entry in group], dtype=float)
        projector_distances = np.array([entry.projector_distance for entry in group], dtype=float)
        overlaps = np.array([entry.smallest_subspace_overlap for entry in group], dtype=float)
        final_entropies = np.array([entry.final_mean_single_qubit_entropy for entry in group], dtype=float)
        max_entropies = np.array([entry.max_mean_single_qubit_entropy for entry in group], dtype=float)
        rows.append(
            {
                "n_qubits": group[0].n_qubits,
                "depth": depth,
                "topology": topology,
                "samples": len(group),
                "hilbert_dimension": hilbert_dimension,
                "layer_orbit_dimension_mean": float(layer_dimensions.mean()),
                "layer_orbit_dimension_std": float(layer_dimensions.std(ddof=0)),
                "layer_full_fraction": float(np.mean(layer_dimensions == hilbert_dimension)),
                "cycle_dimension_mean": float(cycle_dimensions.mean()),
                "cycle_dimension_std": float(cycle_dimensions.std(ddof=0)),
                "cycle_full_fraction": float(np.mean(cycle_dimensions == hilbert_dimension)),
                "projector_distance_mean": float(projector_distances.mean()),
                "projector_distance_std": float(projector_distances.std(ddof=0)),
                "smallest_overlap_mean": float(overlaps.mean()),
                "final_entropy_mean": float(final_entropies.mean()),
                "final_entropy_std": float(final_entropies.std(ddof=0)),
                "max_entropy_mean": float(max_entropies.mean()),
            }
        )
    return rows



def mean_single_qubit_entropy(state: np.ndarray, n_qubits: int) -> float:
    """Average von Neumann entropy of the one-qubit reductions."""
    vector = _normalise(state, 2**n_qubits)
    tensor = vector.reshape((2,) * n_qubits)
    entropies: list[float] = []
    for wire in range(n_qubits):
        moved = np.moveaxis(tensor, wire, 0).reshape(2, -1)
        rho = moved @ moved.conj().T
        eigenvalues = np.linalg.eigvalsh(rho).real
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        entropies.append(float(-np.sum(eigenvalues * np.log2(eigenvalues))))
    return float(np.mean(entropies))


def _entangling_pairs(n_qubits: int, topology: str) -> list[tuple[int, int]]:
    if topology == "none" or n_qubits == 1:
        return []
    if topology == "linear":
        return [(wire, wire + 1) for wire in range(n_qubits - 1)]
    if topology == "ring":
        pairs = [(wire, wire + 1) for wire in range(n_qubits - 1)]
        if n_qubits > 2:
            pairs.append((n_qubits - 1, 0))
        return pairs
    if topology == "full":
        return [(control, target) for control in range(n_qubits) for target in range(control + 1, n_qubits)]
    raise ValueError(f"unknown topology: {topology}")


def _default_state(dimension: int) -> np.ndarray:
    state = np.zeros(dimension, dtype=np.complex128)
    state[0] = 1.0
    return state


def _normalise(vector: np.ndarray, dimension: int) -> np.ndarray:
    state = np.asarray(vector, dtype=np.complex128).reshape(-1)
    if state.size != dimension:
        raise ValueError("initial_state has the wrong dimension")
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("initial_state must be finite and non-zero")
    return state / norm


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("there are no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
