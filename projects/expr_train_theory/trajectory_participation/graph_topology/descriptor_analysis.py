from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_DESCRIPTORS = (
    "n_edges",
    "density",
    "mean_degree",
    "max_degree",
    "algebraic_connectivity",
    "diameter",
    "mean_shortest_path",
)

DEFAULT_METRICS = (
    "d_tp_mean",
    "qfim_rank_mean",
    "qfim_trace_mean",
    "qntk_rank",
    "qntk_effective_rank",
    "qntk_trace",
)


def matched_metric_deltas(rows, *, baseline_topology="none", metrics=DEFAULT_METRICS):
    """Subtract a matched baseline row for each parameter sample.

    Rows are matched by ``parameter_sample``. Graph descriptors are copied from
    the non-baseline row, while every requested metric is reported as a delta
    relative to the baseline topology for the same parameter draw.
    """
    baselines = {
        row["parameter_sample"]: row
        for row in rows
        if row["topology"] == baseline_topology
    }
    if not baselines:
        raise ValueError(f"baseline topology {baseline_topology!r} is missing")

    result = []
    for row in rows:
        sample = row["parameter_sample"]
        if sample not in baselines:
            raise ValueError(f"missing baseline row for parameter sample {sample}")
        baseline = baselines[sample]
        delta_row = {
            "topology": row["topology"],
            "baseline_topology": baseline_topology,
            "parameter_sample": sample,
        }
        for descriptor in DEFAULT_DESCRIPTORS:
            delta_row[descriptor] = row[descriptor]
        for metric in metrics:
            delta_row[f"delta_{metric}"] = float(row[metric] - baseline[metric])
        result.append(delta_row)
    return result


def summarize_matched_deltas(delta_rows, *, metrics=DEFAULT_METRICS):
    groups = {}
    for row in delta_rows:
        groups.setdefault(row["topology"], []).append(row)

    summary = []
    for topology, group in groups.items():
        base = group[0]
        item = {
            "topology": topology,
            "baseline_topology": base["baseline_topology"],
            "parameter_samples": len(group),
        }
        for descriptor in DEFAULT_DESCRIPTORS:
            item[descriptor] = base[descriptor]
        for metric in metrics:
            values = np.asarray([row[f"delta_{metric}"] for row in group], dtype=float)
            item[f"delta_{metric}_mean"] = float(np.mean(values))
            item[f"delta_{metric}_std"] = float(np.std(values))
        summary.append(item)

    return sorted(summary, key=lambda row: (row["n_edges"], row["topology"]))


def _pearson_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return float("nan"), int(x.size)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), int(x.size)
    return float(np.corrcoef(x, y)[0, 1]), int(x.size)


def descriptor_associations(
    summary_rows,
    *,
    baseline_topology="none",
    descriptors=DEFAULT_DESCRIPTORS,
    metrics=DEFAULT_METRICS,
):
    """Compute topology-level Pearson associations with matched metric shifts.

    The baseline topology is excluded. This is deliberately descriptive: with
    only a handful of graph families, these coefficients are not treated as a
    predictive model or as evidence of a monotonic law.
    """
    rows = [row for row in summary_rows if row["topology"] != baseline_topology]
    result = []
    for descriptor in descriptors:
        x = [row[descriptor] for row in rows]
        for metric in metrics:
            target = f"delta_{metric}_mean"
            y = [row[target] for row in rows]
            correlation, n_topologies = _pearson_correlation(x, y)
            result.append(
                {
                    "descriptor": descriptor,
                    "metric": metric,
                    "pearson_r": correlation,
                    "n_topologies": n_topologies,
                }
            )
    return result


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_descriptor_analysis(output_dir, delta_rows, summary_rows, associations, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "graph_topology_matched_deltas_raw.csv", delta_rows)
    _write_csv(output_dir / "graph_topology_matched_deltas_summary.csv", summary_rows)
    _write_csv(output_dir / "graph_topology_descriptor_associations.csv", associations)
    with (output_dir / "graph_topology_descriptor_analysis_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
