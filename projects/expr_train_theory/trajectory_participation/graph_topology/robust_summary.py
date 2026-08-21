from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


EFFECT_CLASSES = ("stable_positive", "stable_negative", "unresolved")


def classify_uncertainty_rows(rows, *, baseline_topology="none"):
    """Classify configuration-level topology effects from bootstrap intervals.

    A configuration is called ``stable_positive`` only when its full bootstrap
    interval lies above zero, and ``stable_negative`` only when the interval
    lies below zero. Intervals overlapping zero stay ``unresolved``. The
    no-entanglement baseline is omitted because its matched delta is defined to
    be zero.
    """
    output = []
    for row in rows:
        if row["topology"] == baseline_topology:
            continue

        relation = row.get("ci_relation")
        if relation == "positive":
            effect_class = "stable_positive"
        elif relation == "negative":
            effect_class = "stable_negative"
        elif relation == "overlaps_zero":
            effect_class = "unresolved"
        else:
            raise ValueError(f"unknown ci_relation: {relation!r}")

        item = dict(row)
        item["effect_class"] = effect_class
        output.append(item)

    return sorted(
        output,
        key=lambda row: (
            row["topology"],
            row["metric"],
            row["n_qubits"],
            row["n_layers"],
        ),
    )


def _resolved_direction(positive, negative):
    if positive and not negative:
        return "positive"
    if negative and not positive:
        return "negative"
    if positive and negative:
        return "mixed"
    return "unresolved"


def _strict_robust_classification(positive, negative, unresolved, total):
    """Return a deliberately conservative across-configuration label."""
    if total > 0 and positive == total:
        return "stable_positive"
    if total > 0 and negative == total:
        return "stable_negative"
    return "unresolved"


def summarize_robust_effects(rows, *, baseline_topology="none"):
    """Summarize configuration-level classifications by topology and metric.

    ``robust_classification`` is intentionally strict: every sampled
    width/depth configuration must have an interval excluding zero in the same
    direction. A mixture of resolved and unresolved configurations, or any
    sign disagreement, is reported as ``unresolved``. The less strict
    ``resolved_direction`` field records whether the configurations that *are*
    resolved agree in sign.
    """
    rows = [row for row in rows if row["topology"] != baseline_topology]
    if not rows:
        return []

    keys = sorted({(row["topology"], row["metric"]) for row in rows})
    output = []
    for topology, metric in keys:
        group = [
            row
            for row in rows
            if row["topology"] == topology and row["metric"] == metric
        ]
        classes = [row["effect_class"] for row in group]
        unknown = sorted(set(classes) - set(EFFECT_CLASSES))
        if unknown:
            raise ValueError(f"unknown effect class(es): {unknown}")

        positive = classes.count("stable_positive")
        negative = classes.count("stable_negative")
        unresolved = classes.count("unresolved")
        total = len(group)
        resolved = positive + negative
        mean_deltas = np.asarray([row["mean_delta"] for row in group], dtype=float)
        mean_deltas = mean_deltas[np.isfinite(mean_deltas)]

        output.append(
            {
                "topology": topology,
                "metric": metric,
                "configurations": total,
                "stable_positive": positive,
                "stable_negative": negative,
                "unresolved": unresolved,
                "stable_positive_fraction": float(positive / total),
                "stable_negative_fraction": float(negative / total),
                "unresolved_fraction": float(unresolved / total),
                "resolved_fraction": float(resolved / total),
                "resolved_direction": _resolved_direction(positive, negative),
                "resolved_sign_consistency": (
                    float(max(positive, negative) / resolved)
                    if resolved
                    else float("nan")
                ),
                "robust_classification": _strict_robust_classification(
                    positive, negative, unresolved, total
                ),
                "mean_delta_across_configurations": (
                    float(np.mean(mean_deltas)) if mean_deltas.size else float("nan")
                ),
            }
        )

    return output


def build_robust_topology_summary(uncertainty_rows, *, baseline_topology="none"):
    """Build configuration-level and aggregate robust-effect tables."""
    classified = classify_uncertainty_rows(
        uncertainty_rows,
        baseline_topology=baseline_topology,
    )
    summary = summarize_robust_effects(
        classified,
        baseline_topology=baseline_topology,
    )
    return classified, summary


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


def save_robust_summary(output_dir, configuration_rows, summary_rows, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "graph_topology_robust_configurations.csv", configuration_rows)
    _write_csv(output_dir / "graph_topology_robust_summary.csv", summary_rows)
    with (output_dir / "graph_topology_robust_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
