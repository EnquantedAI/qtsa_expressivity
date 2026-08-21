from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from .comprehensive import (
    matched_comprehensive_deltas,
    summarize_comprehensive,
    topology_comprehensive_study,
)


STABILITY_METRICS = (
    "delta_d_tp_equal_normalized",
    "delta_d_tp_fs_normalized",
    "delta_qfim_relative_rank_mean",
    "delta_qntk_effective_rank",
)


def iter_scaling_configs(qubits, layers):
    """Yield positive (n_qubits, n_layers) pairs in deterministic order."""
    qubits = tuple(int(value) for value in qubits)
    layers = tuple(int(value) for value in layers)
    if not qubits or not layers:
        raise ValueError("qubits and layers must be non-empty")
    if any(value < 1 for value in qubits):
        raise ValueError("all qubit counts must be positive")
    if any(value < 1 for value in layers):
        raise ValueError("all layer counts must be positive")
    yield from product(qubits, layers)


def _annotate_summary_row(row, *, n_qubits, n_layers, config_seed):
    item = dict(row)
    ceiling = min(n_layers + 1, 2**n_qubits)
    item.update(
        {
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "config_seed": config_seed,
            "trajectory_ceiling": ceiling,
            "d_tp_equal_normalized": row["d_tp_equal_mean"] / ceiling,
            "d_tp_fs_normalized": row["d_tp_fs_mean"] / ceiling,
            "delta_d_tp_equal_normalized": row["delta_d_tp_equal_mean"] / ceiling,
            "delta_d_tp_fs_normalized": row["delta_d_tp_fs_mean"] / ceiling,
        }
    )
    return item


def topology_scaling_sweep(
    *,
    qubits=(3, 4),
    layers=(1, 2, 3),
    topologies=("none", "line", "ring", "star", "complete"),
    parameter_samples=2,
    data_points=3,
    seed=2026,
    step=1e-6,
    baseline_topology="none",
):
    """Run the comprehensive matched topology study across width and depth.

    Matching is preserved within each (n_qubits, n_layers) configuration: the
    same input set and parameter draws are reused for every topology. Different
    width/depth configurations receive deterministic sub-seeds because their
    parameter tensors have different shapes.

    The returned table contains both raw dTP values and dTP normalized by the
    maximum trajectory dimension min(n_layers + 1, 2**n_qubits). The normalized
    columns are the appropriate default when comparing topology shifts across
    different depths.
    """
    if parameter_samples < 1 or data_points < 1:
        raise ValueError("parameter_samples and data_points must be positive")
    if step <= 0:
        raise ValueError("step must be positive")

    rows = []
    for index, (n_qubits, n_layers) in enumerate(iter_scaling_configs(qubits, layers)):
        config_seed = seed + index
        raw = topology_comprehensive_study(
            n_qubits=n_qubits,
            n_layers=n_layers,
            topologies=topologies,
            parameter_samples=parameter_samples,
            data_points=data_points,
            seed=config_seed,
            step=step,
        )
        deltas = matched_comprehensive_deltas(
            raw,
            baseline_topology=baseline_topology,
        )
        summary = summarize_comprehensive(raw, deltas)
        rows.extend(
            _annotate_summary_row(
                row,
                n_qubits=n_qubits,
                n_layers=n_layers,
                config_seed=config_seed,
            )
            for row in summary
        )

    return sorted(
        rows,
        key=lambda row: (
            row["n_qubits"],
            row["n_layers"],
            row["n_edges"],
            row["topology"],
        ),
    )


def _dominant_sign(positive, negative, zero):
    if positive == negative == 0:
        return "zero"
    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    return "mixed"


def summarize_topology_stability(
    rows,
    *,
    baseline_topology="none",
    metrics=STABILITY_METRICS,
    zero_tolerance=1e-10,
):
    """Summarize sign stability of matched topology shifts across configurations.

    This is a descriptive check, not a significance test. For every topology
    and metric it reports the distribution of matched shifts over the sampled
    width/depth configurations and the fraction of non-zero shifts sharing the
    dominant sign.
    """
    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be non-negative")
    if not rows:
        return []

    topologies = sorted({row["topology"] for row in rows if row["topology"] != baseline_topology})
    output = []
    for topology in topologies:
        group = [row for row in rows if row["topology"] == topology]
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            positive = int(np.sum(values > zero_tolerance))
            negative = int(np.sum(values < -zero_tolerance))
            zero = int(values.size - positive - negative)
            nonzero = positive + negative
            consistency = (
                float(max(positive, negative) / nonzero) if nonzero else float("nan")
            )

            output.append(
                {
                    "topology": topology,
                    "metric": metric,
                    "configurations": int(values.size),
                    "mean_delta": float(np.mean(values)),
                    "std_delta": float(np.std(values)),
                    "min_delta": float(np.min(values)),
                    "max_delta": float(np.max(values)),
                    "positive_fraction": float(positive / values.size),
                    "negative_fraction": float(negative / values.size),
                    "zero_fraction": float(zero / values.size),
                    "dominant_sign": _dominant_sign(positive, negative, zero),
                    "nonzero_sign_consistency": consistency,
                }
            )

    return output


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_scaling_results(output_dir, rows, stability, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "graph_topology_scaling_summary.csv", rows)
    _write_csv(output_dir / "graph_topology_scaling_stability.csv", stability)
    with (output_dir / "graph_topology_scaling_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
