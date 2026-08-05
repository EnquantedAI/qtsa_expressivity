from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core import lanczos
from ..dynamics import exact_state
from ..metrics import krylov_entropy, participation_ratio, spread_complexity, state_probabilities


@dataclass(frozen=True)
class StabilitySettings:
    perturbation_scales: tuple[float, ...]
    repeats: int
    times: tuple[float, ...]
    seed: int
    tolerance: float


def _normalise(vector: np.ndarray) -> np.ndarray:
    state = np.asarray(vector, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("state must have a finite, non-zero norm")
    return state / norm


def perturb_hermitian(
    hamiltonian: np.ndarray,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be square")
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("hamiltonian must be Hermitian")
    if scale < 0:
        raise ValueError("scale cannot be negative")
    if scale == 0:
        return matrix.copy()

    random_matrix = rng.normal(size=matrix.shape) + 1j * rng.normal(size=matrix.shape)
    noise = 0.5 * (random_matrix + random_matrix.conj().T)
    noise_norm = float(np.linalg.norm(noise, ord="fro"))
    reference_norm = max(float(np.linalg.norm(matrix, ord="fro")), 1.0)
    if noise_norm == 0.0:
        return matrix.copy()
    return matrix + scale * reference_norm * noise / noise_norm


def perturb_state(
    state: np.ndarray,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    base = _normalise(state)
    if scale < 0:
        raise ValueError("scale cannot be negative")
    if scale == 0:
        return base.copy()

    noise = rng.normal(size=base.size) + 1j * rng.normal(size=base.size)
    noise -= np.vdot(base, noise) * base
    noise_norm = float(np.linalg.norm(noise))
    if noise_norm == 0.0:
        return base.copy()
    return _normalise(base + scale * noise / noise_norm)


def _curve_summary(
    hamiltonian: np.ndarray,
    state: np.ndarray,
    times: tuple[float, ...],
    tolerance: float,
) -> dict[str, float | int]:
    result = lanczos(
        hamiltonian,
        state,
        max_dimension=hamiltonian.shape[0],
        tolerance=tolerance,
    )

    spreads: list[float] = []
    entropies: list[float] = []
    participation: list[float] = []
    for time in times:
        evolved = exact_state(hamiltonian, state, time)
        probabilities = state_probabilities(evolved, result.basis, tolerance=tolerance)
        spreads.append(spread_complexity(probabilities))
        entropies.append(krylov_entropy(probabilities))
        participation.append(participation_ratio(probabilities))

    return {
        "krylov_dimension": int(result.dimension),
        "relative_krylov_dimension": float(result.dimension / hamiltonian.shape[0]),
        "max_spread_complexity": float(max(spreads)),
        "mean_spread_complexity": float(np.mean(spreads)),
        "max_krylov_entropy": float(max(entropies)),
        "mean_krylov_entropy": float(np.mean(entropies)),
        "max_participation_ratio": float(max(participation)),
        "mean_participation_ratio": float(np.mean(participation)),
    }


def run_stability_study(
    hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    perturbation_scales: Iterable[float] = (0.0, 1e-4, 1e-3, 1e-2),
    repeats: int = 10,
    times: Iterable[float] = tuple(np.linspace(0.0, 4.0, 41)),
    seed: int = 2026,
    tolerance: float = 1e-12,
) -> tuple[list[dict[str, object]], list[dict[str, object]], StabilitySettings]:
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    state = _normalise(initial_state)
    scales = tuple(float(value) for value in perturbation_scales)
    selected_times = tuple(float(value) for value in times)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be square")
    if matrix.shape[0] != state.size:
        raise ValueError("hamiltonian and state dimensions do not match")
    if not np.allclose(matrix, matrix.conj().T, atol=10 * tolerance, rtol=0.0):
        raise ValueError("hamiltonian must be Hermitian")
    if not scales or any(value < 0 or not np.isfinite(value) for value in scales):
        raise ValueError("perturbation scales must be finite and non-negative")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not selected_times or not np.all(np.isfinite(selected_times)):
        raise ValueError("times must be a non-empty finite sequence")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for scale in scales:
        for repeat in range(repeats):
            # The zero-scale row is kept deterministic rather than sampled repeatedly.
            local_rng = rng if scale > 0 else np.random.default_rng(seed)
            perturbed_h = perturb_hermitian(matrix, scale, local_rng)
            perturbed_state = perturb_state(state, scale, local_rng)
            values = _curve_summary(perturbed_h, perturbed_state, selected_times, tolerance)
            rows.append(
                {
                    "perturbation_scale": scale,
                    "repeat": repeat,
                    **values,
                }
            )

    metric_names = [
        "krylov_dimension",
        "relative_krylov_dimension",
        "max_spread_complexity",
        "mean_spread_complexity",
        "max_krylov_entropy",
        "mean_krylov_entropy",
        "max_participation_ratio",
        "mean_participation_ratio",
    ]
    summaries: list[dict[str, object]] = []
    for scale in scales:
        selected = [row for row in rows if row["perturbation_scale"] == scale]
        summary: dict[str, object] = {
            "perturbation_scale": scale,
            "repeats": len(selected),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=0))
            summary[f"{metric}_min"] = float(np.min(values))
            summary[f"{metric}_max"] = float(np.max(values))
        summaries.append(summary)

    settings = StabilitySettings(
        perturbation_scales=scales,
        repeats=repeats,
        times=selected_times,
        seed=seed,
        tolerance=tolerance,
    )
    return rows, summaries, settings


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_stability_study(
    output_directory: str | Path,
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    settings: StabilitySettings,
) -> tuple[Path, Path, Path]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    raw_path = destination / "krylov_stability_raw.csv"
    summary_path = destination / "krylov_stability_summary.csv"
    metadata_path = destination / "krylov_stability_metadata.json"

    _write_csv(raw_path, rows)
    _write_csv(summary_path, summaries)
    metadata_path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    return raw_path, summary_path, metadata_path
