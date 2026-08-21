import csv
import json
from pathlib import Path

import numpy as np

from ..core import trajectory_participation_dimension
from ..snapshots import trajectory_snapshots


def perturb_snapshots(states, scale, rng):
    states = np.asarray(states, dtype=complex)
    if states.ndim != 2:
        raise ValueError("states must be a 2D array")
    if scale < 0:
        raise ValueError("scale must be non-negative")

    if scale == 0:
        return states.copy()

    noise = rng.normal(size=states.shape) + 1j * rng.normal(size=states.shape)
    # Remove the component parallel to each snapshot. This makes the perturbation
    # change the projective state rather than mostly changing its norm or phase.
    overlaps = np.sum(states.conj() * noise, axis=1)
    noise = noise - overlaps[:, None] * states
    noise_norms = np.linalg.norm(noise, axis=1)
    valid = noise_norms > 0
    noise[valid] = noise[valid] / noise_norms[valid, None]

    perturbed = states + float(scale) * noise
    perturbed = perturbed / np.linalg.norm(perturbed, axis=1)[:, None]
    return perturbed


def state_overlap_distance(reference, candidate):
    reference = np.asarray(reference, dtype=complex)
    candidate = np.asarray(candidate, dtype=complex)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate must have the same 2D shape")

    overlaps = np.abs(np.sum(reference.conj() * candidate, axis=1)) ** 2
    overlaps = np.clip(overlaps, 0.0, 1.0)
    return float(np.mean(1.0 - overlaps))


def snapshot_perturbation_study(
    states,
    scales=(0.0, 1e-4, 1e-3, 1e-2, 5e-2),
    *,
    repeats=20,
    seed=2026,
):
    states = np.asarray(states, dtype=complex)
    baseline = trajectory_participation_dimension(states)
    rng = np.random.default_rng(seed)

    rows = []
    for scale in scales:
        for repeat in range(int(repeats)):
            perturbed = perturb_snapshots(states, float(scale), rng)
            result = trajectory_participation_dimension(perturbed)
            rows.append(
                {
                    "scale": float(scale),
                    "repeat": int(repeat),
                    "d_tp": float(result.dimension),
                    "delta_d_tp": float(result.dimension - baseline.dimension),
                    "abs_delta_d_tp": float(abs(result.dimension - baseline.dimension)),
                    "numerical_rank": int(result.numerical_rank),
                    "entropy": float(result.entropy),
                    "mean_infidelity": state_overlap_distance(states, perturbed),
                }
            )
    return rows


def parameter_perturbation_study(
    features,
    parameters,
    *,
    n_qubits,
    scales=(0.0, 1e-4, 1e-3, 1e-2, 5e-2),
    repeats=20,
    seed=2026,
    encoding_axis="Y",
    reupload_axis=None,
    entangle=True,
):
    parameters = np.asarray(parameters, dtype=float)
    baseline_states = trajectory_snapshots(
        features,
        parameters,
        n_qubits=n_qubits,
        encoding_axis=encoding_axis,
        reupload_axis=reupload_axis,
        entangle=entangle,
    )
    baseline = trajectory_participation_dimension(baseline_states)
    rng = np.random.default_rng(seed)

    rows = []
    for scale in scales:
        for repeat in range(int(repeats)):
            if scale == 0:
                changed = parameters.copy()
            else:
                changed = parameters + float(scale) * rng.normal(size=parameters.shape)
            states = trajectory_snapshots(
                features,
                changed,
                n_qubits=n_qubits,
                encoding_axis=encoding_axis,
                reupload_axis=reupload_axis,
                entangle=entangle,
            )
            result = trajectory_participation_dimension(states)
            rows.append(
                {
                    "scale": float(scale),
                    "repeat": int(repeat),
                    "d_tp": float(result.dimension),
                    "delta_d_tp": float(result.dimension - baseline.dimension),
                    "abs_delta_d_tp": float(abs(result.dimension - baseline.dimension)),
                    "numerical_rank": int(result.numerical_rank),
                    "entropy": float(result.entropy),
                    "mean_infidelity": state_overlap_distance(baseline_states, states),
                }
            )
    return rows


def summarise_rows(rows):
    if not rows:
        return []
    scales = sorted({row["scale"] for row in rows})
    summary = []
    for scale in scales:
        group = [row for row in rows if row["scale"] == scale]
        summary.append(
            {
                "scale": float(scale),
                "samples": len(group),
                "mean_d_tp": float(np.mean([row["d_tp"] for row in group])),
                "std_d_tp": float(np.std([row["d_tp"] for row in group])),
                "mean_abs_delta_d_tp": float(np.mean([row["abs_delta_d_tp"] for row in group])),
                "mean_infidelity": float(np.mean([row["mean_infidelity"] for row in group])),
                "mean_rank": float(np.mean([row["numerical_rank"] for row in group])),
            }
        )
    return summary


def save_results(output_dir, prefix, rows, summary, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if rows:
        with (output_dir / f"{prefix}_raw.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if summary:
        with (output_dir / f"{prefix}_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    with (output_dir / f"{prefix}_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata or {}, handle, indent=2)
