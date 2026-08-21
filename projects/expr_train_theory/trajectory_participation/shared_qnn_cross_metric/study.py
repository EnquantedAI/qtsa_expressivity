from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import csv
import json

import numpy as np

from projects.expr_train_theory.effective_dimension_checks.fisher_examples import (
    classical_fisher,
    computational_basis_probabilities,
)
from projects.expr_train_theory.qfim.core import compute_pure_state_qfim, diagnose_qfim
from projects.expr_train_theory.trajectory_participation.calibration.metrics import (
    calibrate_trajectory,
)
from projects.expr_train_theory.trajectory_participation.core import (
    trajectory_participation_dimension,
)
from projects.expr_train_theory.trajectory_participation.shared_qnn import (
    shared_qnn_snapshots,
)


@dataclass(frozen=True)
class SweepConfig:
    n_layers: int
    n_qubits: int
    fm_style: str
    reup_style: str | None


def iter_configs(layers, qubits, feature_maps, reupload_styles):
    for n_layers, n_qubits, fm_style, reup_style in product(
        layers, qubits, feature_maps, reupload_styles
    ):
        yield SweepConfig(
            n_layers=int(n_layers),
            n_qubits=int(n_qubits),
            fm_style=str(fm_style),
            reup_style=reup_style,
        )


def _matrix_rank(matrix, *, atol=1e-9, rtol=1e-7):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return 0
    values = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    largest = max(0.0, float(values[-1]))
    threshold = max(atol, rtol * largest)
    return int(np.count_nonzero(values > threshold))


def _rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _correlation(x, y, *, rank=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    if rank:
        x = _rankdata(x)
        y = _rankdata(y)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_trajectory(theta, trajectory_function, *, qfim_step=1e-6, min_probability=1e-12):
    """Calculate dTP, QFIM and computational-basis CFIM for one trajectory."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    snapshots = np.asarray(trajectory_function(theta), dtype=np.complex128)
    if snapshots.ndim != 2 or snapshots.shape[0] == 0:
        raise ValueError("trajectory_function must return a non-empty 2D state array")

    final_state = lambda parameters: np.asarray(
        trajectory_function(parameters), dtype=np.complex128
    )[-1]
    probability_function = lambda parameters: computational_basis_probabilities(
        final_state(parameters)
    )

    tp = trajectory_participation_dimension(snapshots)
    calibration = calibrate_trajectory(snapshots)
    qfim = compute_pure_state_qfim(final_state, theta, step=qfim_step)
    qdiag = diagnose_qfim(qfim)
    cfim = classical_fisher(
        probability_function,
        theta,
        step=qfim_step,
        min_probability=min_probability,
    )
    cfim_rank = _matrix_rank(cfim)

    return {
        "parameter_count": int(theta.size),
        "snapshot_count": int(snapshots.shape[0]),
        "hilbert_dimension": int(snapshots.shape[1]),
        "d_tp": float(tp.dimension),
        "d_tp_rank_fraction": float(calibration.fraction_of_rank),
        "d_tp_ceiling_fraction": float(calibration.fraction_of_ceiling),
        "trajectory_rank": int(tp.numerical_rank),
        "trajectory_entropy": float(tp.entropy),
        "qfim_rank": int(qdiag.numerical_rank),
        "qfim_relative_rank": float(qdiag.relative_rank),
        "qfim_trace": float(qdiag.trace),
        "qfim_condition": float(qdiag.positive_condition_number),
        "cfim_rank": int(cfim_rank),
        "cfim_relative_rank": float(cfim_rank / theta.size if theta.size else 0.0),
        "cfim_trace": float(np.trace(cfim)),
        "rank_gap_qfim_cfim": int(qdiag.numerical_rank - cfim_rank),
    }


def evaluate_config(
    config,
    *,
    samples=3,
    input_size=3,
    seed=2026,
    qfim_step=1e-6,
    min_probability=1e-12,
    snapshot_fn=shared_qnn_snapshots,
):
    if samples < 1:
        raise ValueError("samples must be positive")
    if input_size < 1:
        raise ValueError("input_size must be positive")

    rng = np.random.default_rng(seed)
    rows = []
    weight_shape = (config.n_layers, config.n_qubits, 3)

    for sample in range(samples):
        inputs = rng.uniform(-np.pi, np.pi, size=input_size)
        weights = rng.uniform(-np.pi, np.pi, size=weight_shape)
        theta = weights.reshape(-1)

        def trajectory_function(parameters):
            reshaped = np.asarray(parameters, dtype=float).reshape(weight_shape)
            return snapshot_fn(
                inputs,
                reshaped,
                n_qubits=config.n_qubits,
                fm_style=config.fm_style,
                reup_style=config.reup_style,
            )

        metrics = evaluate_trajectory(
            theta,
            trajectory_function,
            qfim_step=qfim_step,
            min_probability=min_probability,
        )
        rows.append(
            {
                "n_layers": config.n_layers,
                "n_qubits": config.n_qubits,
                "fm_style": config.fm_style,
                "reup_style": config.reup_style or "none",
                "sample": sample,
                "seed": seed,
                "input_size": input_size,
                **metrics,
            }
        )

    return rows


def summarize_rows(rows):
    if not rows:
        return []
    keys = ("n_layers", "n_qubits", "fm_style", "reup_style")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)

    summary = []
    metric_names = (
        "d_tp",
        "d_tp_rank_fraction",
        "d_tp_ceiling_fraction",
        "trajectory_rank",
        "qfim_relative_rank",
        "qfim_trace",
        "cfim_relative_rank",
        "cfim_trace",
        "rank_gap_qfim_cfim",
    )
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
            row["fm_style"],
            row["reup_style"],
        ),
    )


def metric_correlations(rows):
    if not rows:
        return []
    pairs = (
        ("d_tp", "qfim_relative_rank"),
        ("d_tp", "qfim_trace"),
        ("d_tp", "cfim_relative_rank"),
        ("d_tp", "cfim_trace"),
        ("qfim_relative_rank", "cfim_relative_rank"),
        ("qfim_trace", "cfim_trace"),
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
    layers=(1, 2, 3),
    qubits=(1, 2),
    feature_maps=("Y", "zzfm"),
    reupload_styles=(None, "Y"),
    samples=3,
    input_size=3,
    seed=2026,
    qfim_step=1e-6,
    min_probability=1e-12,
    snapshot_fn=shared_qnn_snapshots,
):
    rows = []
    for index, config in enumerate(iter_configs(layers, qubits, feature_maps, reupload_styles)):
        rows.extend(
            evaluate_config(
                config,
                samples=samples,
                input_size=input_size,
                seed=seed + index,
                qfim_step=qfim_step,
                min_probability=min_probability,
                snapshot_fn=snapshot_fn,
            )
        )
    return rows, summarize_rows(rows), metric_correlations(rows)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir, rows, summary, correlations, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "shared_qnn_cross_metric_raw.csv", rows)
    _write_csv(output_dir / "shared_qnn_cross_metric_summary.csv", summary)
    _write_csv(output_dir / "shared_qnn_cross_metric_correlations.csv", correlations)
    (output_dir / "shared_qnn_cross_metric_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
