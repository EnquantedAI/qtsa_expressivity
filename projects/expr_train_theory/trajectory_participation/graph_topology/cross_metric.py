from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from projects.expr_train_theory.qfim.core import compute_pure_state_qfim, diagnose_qfim
from projects.expr_train_theory.trajectory_participation.qntk_cross_metric.qntk import (
    compute_qntk,
    diagnose_qntk,
)

from ..core import trajectory_participation_dimension
from .metrics import graph_metrics, topology_edges
from .study import topology_snapshots


def topology_final_state(features, parameters, *, n_qubits, edges):
    """Return the final state of the small topology-reference circuit."""
    return topology_snapshots(
        features,
        parameters,
        n_qubits=n_qubits,
        edges=edges,
    )[-1]


def z0_expectation(features, parameters, *, n_qubits, edges):
    """Expectation value of Z on the first qubit of the final state."""
    state = topology_final_state(
        features,
        parameters,
        n_qubits=n_qubits,
        edges=edges,
    )
    probabilities = np.abs(state) ** 2
    half = probabilities.size // 2
    return float(np.sum(probabilities[:half]) - np.sum(probabilities[half:]))


def _qfim_diagnostics(features, parameters, *, n_qubits, edges, step):
    shape = np.asarray(parameters, dtype=float).shape
    flat = np.asarray(parameters, dtype=float).reshape(-1)

    def state_function(theta):
        return topology_final_state(
            features,
            np.asarray(theta, dtype=float).reshape(shape),
            n_qubits=n_qubits,
            edges=edges,
        )

    matrix = compute_pure_state_qfim(state_function, flat, step=step)
    return diagnose_qfim(matrix)


def _qntk_diagnostics(inputs, parameters, *, n_qubits, edges, step):
    shape = np.asarray(parameters, dtype=float).shape
    flat = np.asarray(parameters, dtype=float).reshape(-1)

    def model(features, theta):
        return z0_expectation(
            features,
            np.asarray(theta, dtype=float).reshape(shape),
            n_qubits=n_qubits,
            edges=edges,
        )

    kernel = compute_qntk(model, flat, inputs, step=step)
    return diagnose_qntk(kernel)


def topology_cross_metric_study(
    *,
    n_qubits=3,
    n_layers=2,
    topologies=("none", "line", "ring", "star", "complete"),
    parameter_samples=3,
    data_points=5,
    seed=2026,
    step=1e-6,
):
    """Matched dTP/QFIM/QNTK comparison across entanglement topologies.

    The input set and all variational parameter samples are generated once and
    reused for every topology. For each parameter sample, dTP and QFIM are
    averaged over the same input set. QNTK is built on that input set using the
    final-state <Z_0> output.
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

    rows = []
    parameter_count = n_layers * n_qubits * 3

    for topology in topologies:
        edges = topology_edges(n_qubits, topology)
        graph = graph_metrics(n_qubits, edges)

        for sample, parameters in enumerate(parameter_sets):
            d_tp_values = []
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
                d_tp = trajectory_participation_dimension(states)
                d_tp_values.append(float(d_tp.dimension))
                trajectory_ranks.append(int(d_tp.numerical_rank))

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
                    "d_tp_mean": float(np.mean(d_tp_values)),
                    "d_tp_std": float(np.std(d_tp_values)),
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


def summarize_cross_metric(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["topology"], []).append(row)

    summary = []
    for topology, group in groups.items():
        base = group[0]
        summary.append(
            {
                "topology": topology,
                "n_edges": base["n_edges"],
                "density": base["density"],
                "algebraic_connectivity": base["algebraic_connectivity"],
                "parameter_samples": len(group),
                "d_tp_mean": float(np.mean([r["d_tp_mean"] for r in group])),
                "qfim_rank_mean": float(np.mean([r["qfim_rank_mean"] for r in group])),
                "qfim_trace_mean": float(np.mean([r["qfim_trace_mean"] for r in group])),
                "qntk_rank_mean": float(np.mean([r["qntk_rank"] for r in group])),
                "qntk_effective_rank_mean": float(
                    np.mean([r["qntk_effective_rank"] for r in group])
                ),
                "qntk_trace_mean": float(np.mean([r["qntk_trace"] for r in group])),
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


def save_cross_metric_results(output_dir, rows, summary, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "graph_topology_cross_metric_raw.csv", rows)
    _write_csv(output_dir / "graph_topology_cross_metric_summary.csv", summary)
    with (output_dir / "graph_topology_cross_metric_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
