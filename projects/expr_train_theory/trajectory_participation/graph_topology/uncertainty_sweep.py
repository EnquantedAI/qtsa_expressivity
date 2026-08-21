from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..metric_uncertainty.bootstrap import bootstrap_mean
from .scaling_sweep import STABILITY_METRICS, topology_scaling_sweep


UNCERTAINTY_METRICS = STABILITY_METRICS


def repeat_seeds(*, base_seed=2026, repeats=8, stride=10_000):
    """Return deterministic, separated seeds for independent repeat sweeps."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    return tuple(int(base_seed) + index * int(stride) for index in range(repeats))


def topology_scaling_multiseed(
    *,
    qubits=(3, 4),
    layers=(1, 2, 3),
    topologies=("none", "line", "ring", "star", "complete"),
    parameter_samples=2,
    data_points=3,
    base_seed=2026,
    repeats=8,
    seed_stride=10_000,
    step=1e-6,
    baseline_topology="none",
):
    """Repeat the matched width/depth topology sweep over independent seeds.

    Every repeat keeps the existing matching rule inside a width/depth
    configuration: the same inputs and parameter draws are reused for all
    topologies. Repeats use separated top-level seeds, so uncertainty is
    estimated from independent matched experiments rather than from treating
    topology rows inside one experiment as independent observations.
    """
    seeds = repeat_seeds(base_seed=base_seed, repeats=repeats, stride=seed_stride)
    rows = []
    for repeat_index, seed in enumerate(seeds):
        repeat_rows = topology_scaling_sweep(
            qubits=qubits,
            layers=layers,
            topologies=topologies,
            parameter_samples=parameter_samples,
            data_points=data_points,
            seed=seed,
            step=step,
            baseline_topology=baseline_topology,
        )
        for row in repeat_rows:
            item = dict(row)
            item["repeat_index"] = repeat_index
            item["repeat_seed"] = seed
            rows.append(item)
    return rows


def _ci_relation(lower, upper, *, zero_tolerance):
    if lower > zero_tolerance:
        return "positive"
    if upper < -zero_tolerance:
        return "negative"
    return "overlaps_zero"


def bootstrap_matched_delta_uncertainty(
    rows,
    *,
    metrics=UNCERTAINTY_METRICS,
    baseline_topology="none",
    bootstrap_samples=2000,
    confidence=0.95,
    seed=2026,
    zero_tolerance=1e-10,
):
    """Bootstrap repeat-level means of matched topology deltas.

    The bootstrap unit is one independent repeat seed. The returned percentile
    interval is a finite-sample uncertainty diagnostic, not a proof of a
    topology effect. `ci_relation` only records whether the chosen interval is
    wholly above/below zero or overlaps zero.
    """
    rows = list(rows)
    if not rows:
        return []
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be non-negative")

    keys = sorted(
        {
            (row["n_qubits"], row["n_layers"], row["topology"])
            for row in rows
        }
    )
    output = []
    group_index = 0
    for n_qubits, n_layers, topology in keys:
        group = [
            row
            for row in rows
            if row["n_qubits"] == n_qubits
            and row["n_layers"] == n_layers
            and row["topology"] == topology
        ]
        repeat_ids = [row["repeat_index"] for row in group]
        if len(set(repeat_ids)) != len(repeat_ids):
            raise ValueError("each configuration/topology must have one row per repeat")

        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            estimate = bootstrap_mean(
                values,
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                seed=seed + group_index,
            )
            group_index += 1

            positive = int(np.sum(values > zero_tolerance))
            negative = int(np.sum(values < -zero_tolerance))
            zero = int(values.size - positive - negative)
            ci_relation = _ci_relation(
                estimate.lower,
                estimate.upper,
                zero_tolerance=zero_tolerance,
            )

            output.append(
                {
                    "n_qubits": n_qubits,
                    "n_layers": n_layers,
                    "topology": topology,
                    "baseline_topology": baseline_topology,
                    "metric": metric,
                    "repeats": int(values.size),
                    "mean_delta": estimate.estimate,
                    "std_delta": float(np.std(values)),
                    "ci_lower": estimate.lower,
                    "ci_upper": estimate.upper,
                    "confidence": float(confidence),
                    "bootstrap_samples": estimate.bootstrap_samples,
                    "positive_fraction": float(positive / values.size),
                    "negative_fraction": float(negative / values.size),
                    "zero_fraction": float(zero / values.size),
                    "ci_relation": ci_relation,
                    "ci_excludes_zero": ci_relation != "overlaps_zero",
                }
            )

    return output


def summarize_ci_stability(rows, *, baseline_topology="none"):
    """Summarize CI direction across width/depth configurations.

    This is deliberately conservative: configurations whose interval overlaps
    zero are counted separately rather than assigned the sign of their point
    estimate.
    """
    rows = [row for row in rows if row["topology"] != baseline_topology]
    if not rows:
        return []

    keys = sorted({(row["topology"], row["metric"]) for row in rows})
    output = []
    for topology, metric in keys:
        group = [
            row for row in rows if row["topology"] == topology and row["metric"] == metric
        ]
        positive = sum(row["ci_relation"] == "positive" for row in group)
        negative = sum(row["ci_relation"] == "negative" for row in group)
        overlap = sum(row["ci_relation"] == "overlaps_zero" for row in group)
        resolved = positive + negative
        dominant = "mixed"
        if resolved == 0:
            dominant = "unresolved"
        elif positive > 0 and negative == 0:
            dominant = "positive"
        elif negative > 0 and positive == 0:
            dominant = "negative"

        output.append(
            {
                "topology": topology,
                "metric": metric,
                "configurations": len(group),
                "ci_positive": positive,
                "ci_negative": negative,
                "ci_overlaps_zero": overlap,
                "resolved_fraction": float(resolved / len(group)),
                "dominant_resolved_sign": dominant,
                "resolved_sign_consistency": (
                    float(max(positive, negative) / resolved) if resolved else float("nan")
                ),
            }
        )
    return output


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_uncertainty_results(output_dir, repeat_rows, uncertainty_rows, stability_rows, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "graph_topology_multiseed_raw.csv", repeat_rows)
    _write_csv(output_dir / "graph_topology_multiseed_uncertainty.csv", uncertainty_rows)
    _write_csv(output_dir / "graph_topology_multiseed_ci_stability.csv", stability_rows)
    with (output_dir / "graph_topology_multiseed_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
