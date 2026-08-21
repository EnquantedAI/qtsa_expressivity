from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..arc_length_weighting.weights import (
    arc_length_snapshot_weights,
    arc_length_weighted_participation_dimension,
    cumulative_fubini_study_length,
)
from ..core import trajectory_participation_dimension
from .metrics import graph_metrics, topology_edges
from .study import topology_snapshots


def topology_fs_weighting_study(
    *,
    n_qubits=3,
    n_layers=2,
    topologies=("none", "line", "ring", "star", "complete"),
    parameter_samples=3,
    data_points=5,
    seed=2026,
):
    """Matched equal-weight vs Fubini--Study-weighted dTP across topologies.

    Inputs and variational parameter samples are generated once and reused for
    every topology. For each matched trajectory we record both dTP conventions,
    the projective path length, and simple diagnostics of the arc-length weights.
    """
    if n_qubits < 1 or n_layers < 1:
        raise ValueError("n_qubits and n_layers must be positive")
    if parameter_samples < 1 or data_points < 1:
        raise ValueError("parameter_samples and data_points must be positive")

    rng = np.random.default_rng(seed)
    inputs = rng.uniform(-np.pi, np.pi, size=(data_points, n_qubits))
    parameter_sets = rng.uniform(
        -np.pi,
        np.pi,
        size=(parameter_samples, n_layers, n_qubits, 3),
    )

    rows = []
    for topology in topologies:
        edges = topology_edges(n_qubits, topology)
        graph = graph_metrics(n_qubits, edges)

        for sample, parameters in enumerate(parameter_sets):
            equal_values = []
            fs_values = []
            path_lengths = []
            weight_minima = []
            weight_maxima = []
            trajectory_ranks = []

            for features in inputs:
                states = topology_snapshots(
                    features,
                    parameters,
                    n_qubits=n_qubits,
                    edges=edges,
                )
                equal = trajectory_participation_dimension(states)
                fs_value = arc_length_weighted_participation_dimension(states)
                fs_weights = arc_length_snapshot_weights(states)
                cumulative = cumulative_fubini_study_length(states)

                equal_values.append(float(equal.dimension))
                fs_values.append(float(fs_value))
                path_lengths.append(float(cumulative[-1]))
                weight_minima.append(float(np.min(fs_weights)))
                weight_maxima.append(float(np.max(fs_weights)))
                trajectory_ranks.append(int(equal.numerical_rank))

            equal_mean = float(np.mean(equal_values))
            fs_mean = float(np.mean(fs_values))
            rows.append(
                {
                    "topology": topology,
                    "parameter_sample": sample,
                    "n_qubits": n_qubits,
                    "n_layers": n_layers,
                    "data_points": data_points,
                    "n_edges": graph.n_edges,
                    "density": graph.density,
                    "mean_degree": graph.mean_degree,
                    "max_degree": graph.max_degree,
                    "connected_components": graph.connected_components,
                    "diameter": graph.diameter,
                    "mean_shortest_path": graph.mean_shortest_path,
                    "algebraic_connectivity": graph.algebraic_connectivity,
                    "d_tp_equal_mean": equal_mean,
                    "d_tp_fs_mean": fs_mean,
                    "fs_minus_equal_mean": fs_mean - equal_mean,
                    "fs_over_equal_mean": fs_mean / equal_mean,
                    "path_length_fs_mean": float(np.mean(path_lengths)),
                    "fs_weight_min_mean": float(np.mean(weight_minima)),
                    "fs_weight_max_mean": float(np.mean(weight_maxima)),
                    "trajectory_rank_mean": float(np.mean(trajectory_ranks)),
                }
            )

    return rows


def matched_topology_weighting_deltas(rows, *, baseline_topology="none"):
    """Compare topology effects under equal and FS-weighted dTP conventions."""
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
        delta_equal = float(row["d_tp_equal_mean"] - baseline["d_tp_equal_mean"])
        delta_fs = float(row["d_tp_fs_mean"] - baseline["d_tp_fs_mean"])
        result.append(
            {
                "topology": row["topology"],
                "baseline_topology": baseline_topology,
                "parameter_sample": sample,
                "n_edges": row["n_edges"],
                "density": row["density"],
                "algebraic_connectivity": row["algebraic_connectivity"],
                "delta_d_tp_equal": delta_equal,
                "delta_d_tp_fs": delta_fs,
                "delta_fs_minus_equal": delta_fs - delta_equal,
                "delta_path_length_fs": float(
                    row["path_length_fs_mean"] - baseline["path_length_fs_mean"]
                ),
            }
        )
    return result


def summarize_topology_weighting(rows, delta_rows):
    """Summarize raw weighting diagnostics and matched topology shifts."""
    raw_groups = {}
    for row in rows:
        raw_groups.setdefault(row["topology"], []).append(row)
    delta_groups = {}
    for row in delta_rows:
        delta_groups.setdefault(row["topology"], []).append(row)

    summary = []
    for topology, group in raw_groups.items():
        deltas = delta_groups[topology]
        base = group[0]
        summary.append(
            {
                "topology": topology,
                "n_edges": base["n_edges"],
                "density": base["density"],
                "algebraic_connectivity": base["algebraic_connectivity"],
                "parameter_samples": len(group),
                "d_tp_equal_mean": float(np.mean([r["d_tp_equal_mean"] for r in group])),
                "d_tp_fs_mean": float(np.mean([r["d_tp_fs_mean"] for r in group])),
                "fs_minus_equal_mean": float(
                    np.mean([r["fs_minus_equal_mean"] for r in group])
                ),
                "path_length_fs_mean": float(
                    np.mean([r["path_length_fs_mean"] for r in group])
                ),
                "delta_d_tp_equal_mean": float(
                    np.mean([r["delta_d_tp_equal"] for r in deltas])
                ),
                "delta_d_tp_fs_mean": float(
                    np.mean([r["delta_d_tp_fs"] for r in deltas])
                ),
                "delta_fs_minus_equal_mean": float(
                    np.mean([r["delta_fs_minus_equal"] for r in deltas])
                ),
                "delta_path_length_fs_mean": float(
                    np.mean([r["delta_path_length_fs"] for r in deltas])
                ),
            }
        )

    return sorted(summary, key=lambda row: (row["n_edges"], row["topology"]))


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_topology_weighting_results(output_dir, rows, delta_rows, summary, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "graph_topology_fs_weighting_raw.csv", rows)
    _write_csv(output_dir / "graph_topology_fs_weighting_deltas.csv", delta_rows)
    _write_csv(output_dir / "graph_topology_fs_weighting_summary.csv", summary)
    with (output_dir / "graph_topology_fs_weighting_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
