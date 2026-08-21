import csv
import json
from pathlib import Path

import numpy as np

from projects.expr_train_theory.krylov.circuit_orbits.circuits import (
    cnot,
    embed_one_qubit,
    identity,
    rotation_x,
    rotation_y,
    rotation_z,
)

from ..core import trajectory_participation_dimension
from .metrics import graph_metrics, topology_edges


def _angle_encode(features, n_qubits):
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0
    unitary = identity(n_qubits)
    for wire, value in enumerate(np.asarray(features, dtype=float)[:n_qubits]):
        unitary = embed_one_qubit(rotation_y(value), wire, n_qubits) @ unitary
    return unitary @ state


def topology_snapshots(features, parameters, *, n_qubits, edges):
    """Small dense reference circuit used only for topology comparisons."""
    params = np.asarray(parameters, dtype=float)
    if params.ndim != 3 or params.shape[1:] != (n_qubits, 3):
        raise ValueError("parameters must have shape (layers, n_qubits, 3)")

    state = _angle_encode(features, n_qubits)
    snapshots = [state.copy()]
    entanglers = tuple(edges)

    for layer_params in params:
        layer = identity(n_qubits)
        for wire, (rx, ry, rz) in enumerate(layer_params):
            rotation = rotation_z(rz) @ rotation_y(ry) @ rotation_x(rx)
            layer = embed_one_qubit(rotation, wire, n_qubits) @ layer
        for control, target in entanglers:
            layer = cnot(control, target, n_qubits) @ layer
        state = layer @ state
        snapshots.append(state.copy())

    return np.asarray(snapshots)


def topology_study(
    *,
    n_qubits=4,
    n_layers=4,
    topologies=("none", "line", "ring", "star", "complete"),
    samples=8,
    seed=2026,
):
    if n_qubits < 1 or n_layers < 1 or samples < 1:
        raise ValueError("n_qubits, n_layers and samples must be positive")

    rng = np.random.default_rng(seed)
    matched_inputs = [rng.uniform(-np.pi, np.pi, size=n_qubits) for _ in range(samples)]
    matched_parameters = [
        rng.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 3)) for _ in range(samples)
    ]

    rows = []
    for topology in topologies:
        edges = topology_edges(n_qubits, topology)
        g = graph_metrics(n_qubits, edges)
        for sample, (features, parameters) in enumerate(zip(matched_inputs, matched_parameters)):
            states = topology_snapshots(
                features, parameters, n_qubits=n_qubits, edges=edges
            )
            result = trajectory_participation_dimension(states)
            rows.append(
                {
                    "topology": topology,
                    "sample": sample,
                    "n_qubits": n_qubits,
                    "n_layers": n_layers,
                    "n_edges": g.n_edges,
                    "density": g.density,
                    "mean_degree": g.mean_degree,
                    "max_degree": g.max_degree,
                    "connected_components": g.connected_components,
                    "diameter": g.diameter,
                    "mean_shortest_path": g.mean_shortest_path,
                    "algebraic_connectivity": g.algebraic_connectivity,
                    "d_tp": float(result.dimension),
                    "d_tp_normalized": float(result.dimension / min(states.shape)),
                    "trajectory_rank": int(result.numerical_rank),
                }
            )
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["topology"], []).append(row)
    result = []
    for topology, group in groups.items():
        d_tp = np.asarray([r["d_tp"] for r in group], dtype=float)
        base = group[0]
        result.append(
            {
                "topology": topology,
                "n_edges": base["n_edges"],
                "density": base["density"],
                "mean_degree": base["mean_degree"],
                "diameter": base["diameter"],
                "algebraic_connectivity": base["algebraic_connectivity"],
                "samples": len(group),
                "d_tp_mean": float(np.mean(d_tp)),
                "d_tp_std": float(np.std(d_tp)),
            }
        )
    return sorted(result, key=lambda row: (row["n_edges"], row["topology"]))


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir, rows, summary, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "graph_topology_raw.csv", rows)
    _write_csv(output_dir / "graph_topology_summary.csv", summary)
    with (output_dir / "graph_topology_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
