from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import numpy as np

from projects.expr_train_theory.trajectory_participation.core import (
    trajectory_participation_dimension,
)
from projects.expr_train_theory.trajectory_participation.shared_qnn import (
    native_shared_qnn_snapshots,
    reference_shared_qnn_snapshots,
)


@dataclass(frozen=True)
class ParityConfig:
    n_layers: int
    n_qubits: int
    fm_style: str
    reup_style: str | None = None
    input_size: int = 2

    def __post_init__(self):
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if self.input_size < 1:
            raise ValueError("input_size must be positive")


# A compact matrix that exercises every feature-map family used by the shared
# model, all supported reupload axes, more than one width and more than one
# depth.  It is intentionally a parity suite, not an architecture benchmark.
DEFAULT_PARITY_CONFIGS = (
    ParityConfig(1, 2, "Y", None, 2),
    ParityConfig(2, 2, "Y", "X", 2),
    ParityConfig(3, 3, "Y", "Z", 2),
    ParityConfig(2, 3, "X", "Y", 3),
    ParityConfig(2, 2, "Z", None, 2),
    ParityConfig(2, 2, "zzfm", "Y", 2),
    ParityConfig(2, 3, "zzfm", None, 2),
    ParityConfig(2, 3, "iqp", "X", 2),
)


def _as_state(state):
    values = np.asarray(state, dtype=np.complex128).reshape(-1)
    if values.size == 0:
        raise ValueError("state cannot be empty")
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("state must have a finite non-zero norm")
    return values / norm


def state_fidelity(first, second):
    """Pure-state fidelity, insensitive to a global phase."""
    first = _as_state(first)
    second = _as_state(second)
    if first.shape != second.shape:
        raise ValueError("states must have the same dimension")
    return float(np.abs(np.vdot(first, second)) ** 2)


def phase_aligned_state_error(candidate, reference):
    """Maximum state-vector error after removing one global phase."""
    candidate = _as_state(candidate)
    reference = _as_state(reference)
    if candidate.shape != reference.shape:
        raise ValueError("states must have the same dimension")

    overlap = np.vdot(reference, candidate)
    if np.abs(overlap) > 0.0:
        candidate = candidate * np.exp(-1j * np.angle(overlap))
    return float(np.max(np.abs(candidate - reference)))


def compare_trajectories(native, reference, *, tolerance=1e-8):
    """Compare native and reference trajectories snapshot by snapshot.

    State comparisons are global-phase invariant.  The report also checks the
    trajectory participation dimension, which is invariant under an
    independent global phase for every snapshot.
    """
    native = np.asarray(native, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    if native.ndim != 2 or reference.ndim != 2:
        raise ValueError("trajectories must be 2D state arrays")
    if native.shape != reference.shape:
        raise ValueError("native and reference trajectories must have the same shape")
    if native.shape[0] == 0:
        raise ValueError("trajectory cannot be empty")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    fidelities = np.asarray(
        [state_fidelity(a, b) for a, b in zip(native, reference)], dtype=float
    )
    aligned_errors = np.asarray(
        [phase_aligned_state_error(a, b) for a, b in zip(native, reference)],
        dtype=float,
    )
    native_tp = trajectory_participation_dimension(native)
    reference_tp = trajectory_participation_dimension(reference)
    d_tp_error = abs(native_tp.dimension - reference_tp.dimension)

    native_gram = native @ native.conj().T
    reference_gram = reference @ reference.conj().T
    gram_magnitude_error = float(
        np.max(np.abs(np.abs(native_gram) - np.abs(reference_gram)))
    )

    max_infidelity = float(np.max(1.0 - fidelities))
    max_aligned_error = float(np.max(aligned_errors))
    passed = bool(
        max_infidelity <= tolerance
        and max_aligned_error <= tolerance
        and d_tp_error <= tolerance
        and gram_magnitude_error <= tolerance
    )

    return {
        "snapshot_count": int(native.shape[0]),
        "hilbert_dimension": int(native.shape[1]),
        "min_snapshot_fidelity": float(np.min(fidelities)),
        "max_snapshot_infidelity": max_infidelity,
        "max_phase_aligned_state_error": max_aligned_error,
        "max_gram_magnitude_error": gram_magnitude_error,
        "native_d_tp": float(native_tp.dimension),
        "reference_d_tp": float(reference_tp.dimension),
        "d_tp_abs_error": float(d_tp_error),
        "native_trajectory_rank": int(native_tp.numerical_rank),
        "reference_trajectory_rank": int(reference_tp.numerical_rank),
        "passed": passed,
    }


def evaluate_config(
    config,
    *,
    samples=2,
    seed=2026,
    tolerance=1e-8,
    native_fn=native_shared_qnn_snapshots,
    reference_fn=reference_shared_qnn_snapshots,
):
    """Run matched native/reference comparisons for one architecture."""
    if samples < 1:
        raise ValueError("samples must be positive")

    rng = np.random.default_rng(seed)
    rows = []
    weight_shape = (config.n_layers, config.n_qubits, 3)

    for sample in range(samples):
        inputs = rng.uniform(-np.pi, np.pi, size=config.input_size)
        weights = rng.uniform(-np.pi, np.pi, size=weight_shape)

        native = native_fn(
            inputs,
            weights,
            n_qubits=config.n_qubits,
            fm_style=config.fm_style,
            reup_style=config.reup_style,
        )
        reference = reference_fn(
            inputs,
            weights,
            n_qubits=config.n_qubits,
            fm_style=config.fm_style,
            reup_style=config.reup_style,
        )
        comparison = compare_trajectories(native, reference, tolerance=tolerance)
        rows.append(
            {
                "n_layers": config.n_layers,
                "n_qubits": config.n_qubits,
                "fm_style": config.fm_style,
                "reup_style": config.reup_style or "none",
                "input_size": config.input_size,
                "sample": sample,
                "seed": seed,
                "tolerance": tolerance,
                **comparison,
            }
        )

    return rows


def summarize_rows(rows):
    if not rows:
        return []

    keys = ("n_layers", "n_qubits", "fm_style", "reup_style", "input_size")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)

    summary = []
    for key, group in groups.items():
        record = {**dict(zip(keys, key)), "samples": len(group)}
        record["all_passed"] = bool(all(row["passed"] for row in group))
        record["min_snapshot_fidelity"] = float(
            min(row["min_snapshot_fidelity"] for row in group)
        )
        record["max_snapshot_infidelity"] = float(
            max(row["max_snapshot_infidelity"] for row in group)
        )
        record["max_phase_aligned_state_error"] = float(
            max(row["max_phase_aligned_state_error"] for row in group)
        )
        record["max_gram_magnitude_error"] = float(
            max(row["max_gram_magnitude_error"] for row in group)
        )
        record["max_d_tp_abs_error"] = float(
            max(row["d_tp_abs_error"] for row in group)
        )
        summary.append(record)

    return sorted(
        summary,
        key=lambda row: (
            row["n_qubits"],
            row["n_layers"],
            row["fm_style"],
            row["reup_style"],
            row["input_size"],
        ),
    )


def run_parity_study(
    *,
    configs=DEFAULT_PARITY_CONFIGS,
    samples=2,
    seed=2026,
    tolerance=1e-8,
    native_fn=native_shared_qnn_snapshots,
    reference_fn=reference_shared_qnn_snapshots,
):
    rows = []
    for index, config in enumerate(configs):
        rows.extend(
            evaluate_config(
                config,
                samples=samples,
                seed=seed + index,
                tolerance=tolerance,
                native_fn=native_fn,
                reference_fn=reference_fn,
            )
        )
    return rows, summarize_rows(rows)


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir, rows, summary, *, configs, samples, seed, tolerance):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "shared_qnn_parity_raw.csv", rows)
    _write_csv(output_dir / "shared_qnn_parity_summary.csv", summary)
    metadata = {
        "samples": samples,
        "seed": seed,
        "tolerance": tolerance,
        "configs": [asdict(config) for config in configs],
        "comparison": "native src.models.gqnn snapshots vs reconstructed reference path",
        "state_comparison": "global-phase invariant",
    }
    (output_dir / "shared_qnn_parity_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
