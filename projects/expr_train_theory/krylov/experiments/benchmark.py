from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core import lanczos
from ..dynamics import exact_state, projected_state
from ..metrics import krylov_entropy, participation_ratio, spread_complexity, state_probabilities
from ..models import basis_state, path_hamiltonian, two_qubit_ising


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    hamiltonian: np.ndarray
    initial_state: np.ndarray
    description: str


@dataclass(frozen=True)
class BenchmarkSettings:
    times: tuple[float, ...]
    tolerance: float


def _normalised(vector: np.ndarray) -> np.ndarray:
    state = np.asarray(vector, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(state)
    if norm == 0.0:
        raise ValueError("initial state must be non-zero")
    return state / norm


def default_cases() -> list[BenchmarkCase]:
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    return [
        BenchmarkCase(
            name="path4_edge",
            hamiltonian=path_hamiltonian(4),
            initial_state=basis_state(0, 4),
            description="Four-site path, state localised at an edge.",
        ),
        BenchmarkCase(
            name="path4_middle",
            hamiltonian=path_hamiltonian(4),
            initial_state=basis_state(1, 4),
            description="Four-site path, state localised at an inner site.",
        ),
        BenchmarkCase(
            name="ising_00",
            hamiltonian=two_qubit_ising(),
            initial_state=basis_state(0, 4),
            description="Two-qubit Ising example, computational state |00>.",
        ),
        BenchmarkCase(
            name="ising_plus_plus",
            hamiltonian=two_qubit_ising(),
            initial_state=np.kron(plus, plus),
            description="Two-qubit Ising example, product state |++>.",
        ),
        BenchmarkCase(
            name="diagonal_eigenstate",
            hamiltonian=np.diag([0.0, 1.0, 2.0, 3.0]).astype(np.complex128),
            initial_state=basis_state(2, 4),
            description="Diagonal Hamiltonian started from an eigenstate.",
        ),
    ]


def run_benchmark(
    cases: Iterable[BenchmarkCase] | None = None,
    times: Iterable[float] | None = None,
    tolerance: float = 1e-12,
) -> tuple[list[dict[str, object]], list[dict[str, object]], BenchmarkSettings]:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    selected_cases = list(default_cases() if cases is None else cases)
    if not selected_cases:
        raise ValueError("at least one benchmark case is required")

    selected_times = tuple(float(value) for value in (np.linspace(0.0, 4.0, 41) if times is None else times))
    if not selected_times or not np.all(np.isfinite(selected_times)):
        raise ValueError("times must be a non-empty finite sequence")

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for case in selected_cases:
        hamiltonian = np.asarray(case.hamiltonian, dtype=np.complex128)
        initial_state = _normalised(case.initial_state)
        result = lanczos(
            hamiltonian,
            initial_state,
            max_dimension=hamiltonian.shape[0],
            tolerance=tolerance,
        )

        case_rows: list[dict[str, object]] = []
        for time in selected_times:
            exact = exact_state(hamiltonian, initial_state, time)
            projected = projected_state(result, time)
            probabilities = state_probabilities(exact, result.basis, tolerance=tolerance)
            projection_weight = float(probabilities.sum())
            state_error = float(np.linalg.norm(exact - projected))

            row = {
                "case": case.name,
                "time": time,
                "hilbert_dimension": int(hamiltonian.shape[0]),
                "krylov_dimension": int(result.dimension),
                "relative_krylov_dimension": float(result.dimension / hamiltonian.shape[0]),
                "spread_complexity": spread_complexity(probabilities),
                "krylov_entropy": krylov_entropy(probabilities),
                "participation_ratio": participation_ratio(probabilities),
                "projection_weight": projection_weight,
                "state_error": state_error,
            }
            rows.append(row)
            case_rows.append(row)

        summaries.append(
            {
                "case": case.name,
                "description": case.description,
                "hilbert_dimension": int(hamiltonian.shape[0]),
                "krylov_dimension": int(result.dimension),
                "relative_krylov_dimension": float(result.dimension / hamiltonian.shape[0]),
                "max_spread_complexity": max(float(row["spread_complexity"]) for row in case_rows),
                "mean_spread_complexity": float(np.mean([row["spread_complexity"] for row in case_rows])),
                "max_krylov_entropy": max(float(row["krylov_entropy"]) for row in case_rows),
                "max_participation_ratio": max(float(row["participation_ratio"]) for row in case_rows),
                "max_state_error": max(float(row["state_error"]) for row in case_rows),
                "min_projection_weight": min(float(row["projection_weight"]) for row in case_rows),
                "lanczos_residual_norm": float(result.residual_norm),
                "alpha": json.dumps(result.alpha.tolist()),
                "beta": json.dumps(result.beta.tolist()),
            }
        )

    return rows, summaries, BenchmarkSettings(times=selected_times, tolerance=tolerance)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_benchmark(
    output_directory: str | Path,
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    settings: BenchmarkSettings,
) -> tuple[Path, Path, Path]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    raw_path = destination / "krylov_benchmark_raw.csv"
    summary_path = destination / "krylov_benchmark_summary.csv"
    metadata_path = destination / "krylov_benchmark_metadata.json"

    _write_csv(raw_path, rows)
    _write_csv(summary_path, summaries)
    metadata_path.write_text(
        json.dumps(asdict(settings), indent=2),
        encoding="utf-8",
    )
    return raw_path, summary_path, metadata_path
