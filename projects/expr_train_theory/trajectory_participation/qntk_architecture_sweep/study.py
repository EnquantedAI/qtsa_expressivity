from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import csv
import json

import numpy as np

from projects.expr_train_theory.trajectory_participation.core import (
    trajectory_participation_dimension,
)
from projects.expr_train_theory.trajectory_participation.qntk_cross_metric.qntk import (
    compute_qntk,
    diagnose_qntk,
)
from projects.expr_train_theory.trajectory_participation.snapshots import (
    trajectory_snapshots,
)


@dataclass(frozen=True)
class SweepConfig:
    n_layers: int
    n_qubits: int
    reupload_axis: str | None
    entangle: bool


def iter_configs(layers, qubits, reupload_axes, entangling):
    for n_layers, n_qubits, reupload_axis, entangle in product(
        layers, qubits, reupload_axes, entangling
    ):
        yield SweepConfig(
            n_layers=int(n_layers),
            n_qubits=int(n_qubits),
            reupload_axis=reupload_axis,
            entangle=bool(entangle),
        )


def _z_expectation(state, wire, n_qubits):
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    if state.size != 2**n_qubits:
        raise ValueError("state has the wrong dimension")
    probabilities = np.abs(state) ** 2
    shift = n_qubits - wire - 1
    signs = 1.0 - 2.0 * ((np.arange(state.size) >> shift) & 1)
    return float(np.sum(signs * probabilities))


def _trajectory_for(x, theta, config):
    params = np.asarray(theta, dtype=float).reshape(
        config.n_layers, config.n_qubits, 3
    )
    return trajectory_snapshots(
        x,
        params,
        n_qubits=config.n_qubits,
        encoding_axis="Y",
        reupload_axis=config.reupload_axis,
        entangle=config.entangle,
    )


def _model_output(x, theta, config):
    states = _trajectory_for(x, theta, config)
    return _z_expectation(states[-1], 0, config.n_qubits)


def evaluate_config(
    config,
    *,
    samples=4,
    dataset_size=5,
    seed=2026,
    qntk_step=1e-6,
):
    if samples < 1:
        raise ValueError("samples must be positive")
    if dataset_size < 2:
        raise ValueError("dataset_size must be at least two")

    rng = np.random.default_rng(seed)
    rows = []
    parameter_count = config.n_layers * config.n_qubits * 3

    for sample in range(samples):
        X = rng.uniform(-np.pi, np.pi, size=(dataset_size, config.n_qubits))
        theta = rng.uniform(-np.pi, np.pi, size=parameter_count)

        d_tp_values = []
        ranks = []
        for x in X:
            trajectory = _trajectory_for(x, theta, config)
            result = trajectory_participation_dimension(trajectory)
            ceiling = min(trajectory.shape)
            d_tp_values.append(float(result.dimension / ceiling))
            ranks.append(int(result.numerical_rank))

        model = lambda x, parameters: _model_output(x, parameters, config)
        kernel = compute_qntk(model, theta, X, step=qntk_step)
        qdiag = diagnose_qntk(kernel)

        rows.append(
            {
                "n_layers": config.n_layers,
                "n_qubits": config.n_qubits,
                "reupload_axis": config.reupload_axis or "none",
                "entangle": config.entangle,
                "sample": sample,
                "seed": seed,
                "dataset_size": dataset_size,
                "parameter_count": parameter_count,
                "d_tp_normalized_mean": float(np.mean(d_tp_values)),
                "d_tp_normalized_std": float(np.std(d_tp_values)),
                "trajectory_rank_mean": float(np.mean(ranks)),
                "qntk_rank": qdiag.rank,
                "qntk_trace": qdiag.trace,
                "qntk_effective_rank": qdiag.effective_rank,
                "qntk_element_variance": qdiag.element_variance,
                "qntk_condition": qdiag.condition_number,
            }
        )

    return rows


def summarize_rows(rows):
    if not rows:
        return []

    keys = ("n_layers", "n_qubits", "reupload_axis", "entangle")
    metric_names = (
        "d_tp_normalized_mean",
        "trajectory_rank_mean",
        "qntk_rank",
        "qntk_trace",
        "qntk_effective_rank",
        "qntk_element_variance",
    )
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)

    summary = []
    for key, group in groups.items():
        record = {**dict(zip(keys, key)), "samples": len(group)}
        for name in metric_names:
            values = np.asarray([row[name] for row in group], dtype=float)
            record[f"{name}_mean"] = float(np.mean(values))
            record[f"{name}_std"] = float(np.std(values))
        summary.append(record)

    return sorted(
        summary,
        key=lambda row: (
            row["n_qubits"],
            row["n_layers"],
            row["reupload_axis"],
            row["entangle"],
        ),
    )


def _rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _correlation(x, y, *, rank=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    if rank:
        x = _rankdata(x)
        y = _rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def metric_correlations(rows):
    if not rows:
        return []
    pairs = (
        ("d_tp_normalized_mean", "qntk_effective_rank"),
        ("d_tp_normalized_mean", "qntk_trace"),
        ("d_tp_normalized_mean", "qntk_element_variance"),
        ("trajectory_rank_mean", "qntk_rank"),
    )
    output = []
    for left, right in pairs:
        x = [row[left] for row in rows]
        y = [row[right] for row in rows]
        output.append(
            {
                "left": left,
                "right": right,
                "pearson": _correlation(x, y),
                "spearman": _correlation(x, y, rank=True),
                "samples": len(rows),
            }
        )
    return output


def run_sweep(
    *,
    layers=(1, 2, 3, 4),
    qubits=(1, 2, 3),
    reupload_axes=(None, "Y"),
    entangling=(False, True),
    samples=4,
    dataset_size=5,
    seed=2026,
    qntk_step=1e-6,
):
    rows = []
    for index, config in enumerate(
        iter_configs(layers, qubits, reupload_axes, entangling)
    ):
        rows.extend(
            evaluate_config(
                config,
                samples=samples,
                dataset_size=dataset_size,
                seed=seed + index,
                qntk_step=qntk_step,
            )
        )
    return rows, summarize_rows(rows), metric_correlations(rows)


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir, rows, summary, correlations, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "qntk_architecture_raw.csv", rows)
    _write_csv(output_dir / "qntk_architecture_summary.csv", summary)
    _write_csv(output_dir / "qntk_architecture_correlations.csv", correlations)
    (output_dir / "qntk_architecture_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
