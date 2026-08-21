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
from .cross_metric import _qfim_diagnostics, _qntk_diagnostics
from .metrics import graph_metrics, topology_edges
from .study import topology_snapshots


DELTA_METRICS = (
    "d_tp_equal_mean",
    "d_tp_fs_mean",
    "path_length_fs_mean",
    "trajectory_rank_mean",
    "qfim_rank_mean",
    "qfim_relative_rank_mean",
    "qfim_trace_mean",
    "qntk_rank",
    "qntk_effective_rank",
    "qntk_trace",
)

GRAPH_DESCRIPTORS = (
    "n_edges",
    "density",
    "mean_degree",
    "max_degree",
    "connected_components",
    "diameter",
    "mean_shortest_path",
    "algebraic_connectivity",
)


def topology_comprehensive_study(
    *,
    n_qubits=4,
    n_layers=2,
    topologies=("none", "line", "ring", "star", "complete"),
    parameter_samples=3,
    data_points=4,
    seed=2026,
    step=1e-6,
):
    """Collect the main topology diagnostics in one matched experiment.

    The input set and parameter samples are generated once and reused for every
    topology. For each parameter draw the same state trajectories are used for
    equal-weight dTP, Fubini--Study-weighted dTP, projective path length and
    QFIM diagnostics. QNTK is evaluated on the same input set from the final
    state expectation value <Z_0>.
    """
    if n_qubits < 1 or n_layers < 1:
        raise ValueError("n_qubits and n_layers must be positive")
    if parameter_samples < 1 or data_points < 1:
        raise ValueError("parameter_samples and data_points must be positive")
    if step <= 0:
        raise ValueError("step must be positive")

    rng = np.random.default_rng(seed)
    inputs = rng.uniform(-np.pi, np.pi, size=(data_points, n_qubits))
    parameter_sets = rng.uniform(
        -np.pi,
        np.pi,
        size=(parameter_samples, n_layers, n_qubits, 3),
    )
    parameter_count = n_layers * n_qubits * 3

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
            qfim_ranks = []
            qfim_relative_ranks = []
            qfim_traces = []

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

                qfim = _qfim_diagnostics(
                    features,
                    parameters,
                    n_qubits=n_qubits,
                    edges=edges,
                    step=step,
                )
                qfim_ranks.append(qfim.numerical_rank)
                qfim_relative_ranks.append(qfim.relative_rank)
                qfim_traces.append(qfim.trace)

            qntk = _qntk_diagnostics(
                inputs,
                parameters,
                n_qubits=n_qubits,
                edges=edges,
                step=step,
            )

            equal_mean = float(np.mean(equal_values))
            fs_mean = float(np.mean(fs_values))
            rows.append(
                {
                    "topology": topology,
                    "parameter_sample": sample,
                    "n_qubits": n_qubits,
                    "n_layers": n_layers,
                    "data_points": data_points,
                    "parameter_count": parameter_count,
                    "n_edges": graph.n_edges,
                    "density": graph.density,
                    "mean_degree": graph.mean_degree,
                    "max_degree": graph.max_degree,
                    "connected_components": graph.connected_components,
                    "diameter": graph.diameter,
                    "mean_shortest_path": graph.mean_shortest_path,
                    "algebraic_connectivity": graph.algebraic_connectivity,
                    "d_tp_equal_mean": equal_mean,
                    "d_tp_equal_std": float(np.std(equal_values)),
                    "d_tp_fs_mean": fs_mean,
                    "d_tp_fs_std": float(np.std(fs_values)),
                    "fs_minus_equal_mean": fs_mean - equal_mean,
                    "path_length_fs_mean": float(np.mean(path_lengths)),
                    "path_length_fs_std": float(np.std(path_lengths)),
                    "fs_weight_min_mean": float(np.mean(weight_minima)),
                    "fs_weight_max_mean": float(np.mean(weight_maxima)),
                    "trajectory_rank_mean": float(np.mean(trajectory_ranks)),
                    "qfim_rank_mean": float(np.mean(qfim_ranks)),
                    "qfim_relative_rank_mean": float(np.mean(qfim_relative_ranks)),
                    "qfim_trace_mean": float(np.mean(qfim_traces)),
                    "qntk_rank": qntk.rank,
                    "qntk_effective_rank": qntk.effective_rank,
                    "qntk_trace": qntk.trace,
                    "qntk_largest_eigenvalue": qntk.largest_eigenvalue,
                    "qntk_condition": qntk.condition_number,
                }
            )

    return rows


def matched_comprehensive_deltas(
    rows,
    *,
    baseline_topology="none",
    metrics=DELTA_METRICS,
):
    """Subtract the matched baseline for every parameter draw."""
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

        item = {
            "topology": row["topology"],
            "baseline_topology": baseline_topology,
            "parameter_sample": sample,
        }
        for descriptor in GRAPH_DESCRIPTORS:
            item[descriptor] = row[descriptor]
        for metric in metrics:
            item[f"delta_{metric}"] = float(row[metric] - baseline[metric])

        item["delta_fs_minus_equal"] = (
            item["delta_d_tp_fs_mean"] - item["delta_d_tp_equal_mean"]
        )
        result.append(item)

    return result


def summarize_comprehensive(rows, delta_rows, *, metrics=DELTA_METRICS):
    """Return one topology-level table with raw means and matched shifts."""
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
        item = {
            "topology": topology,
            "baseline_topology": deltas[0]["baseline_topology"],
            "parameter_samples": len(group),
        }
        for descriptor in GRAPH_DESCRIPTORS:
            item[descriptor] = base[descriptor]

        raw_mean_metrics = (
            "d_tp_equal_mean",
            "d_tp_fs_mean",
            "fs_minus_equal_mean",
            "path_length_fs_mean",
            "trajectory_rank_mean",
            "qfim_rank_mean",
            "qfim_relative_rank_mean",
            "qfim_trace_mean",
            "qntk_rank",
            "qntk_effective_rank",
            "qntk_trace",
        )
        for metric in raw_mean_metrics:
            item[metric] = float(np.mean([row[metric] for row in group]))

        for metric in metrics:
            values = np.asarray(
                [row[f"delta_{metric}"] for row in deltas],
                dtype=float,
            )
            item[f"delta_{metric}"] = float(np.mean(values))
            item[f"delta_{metric}_std"] = float(np.std(values))

        item["delta_fs_minus_equal"] = float(
            np.mean([row["delta_fs_minus_equal"] for row in deltas])
        )
        summary.append(item)

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


def save_comprehensive_results(output_dir, rows, delta_rows, summary, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "graph_topology_comprehensive_raw.csv", rows)
    _write_csv(output_dir / "graph_topology_comprehensive_deltas.csv", delta_rows)
    _write_csv(output_dir / "graph_topology_comprehensive_summary.csv", summary)
    with (output_dir / "graph_topology_comprehensive_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
