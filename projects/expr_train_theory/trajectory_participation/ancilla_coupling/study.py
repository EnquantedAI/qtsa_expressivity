from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from ..arc_length_weighting.weights import (
    arc_length_weighted_participation_dimension,
    cumulative_fubini_study_length,
)
from ..core import trajectory_participation_dimension
from ..native_shared_qnn_study.study import (
    NativeArchitectureConfig,
    _sample_matched_inputs_and_weights,
    _validate_grid,
    iter_configs,
)
from ..shared_qnn import native_shared_qnn_snapshots
from .metrics import ancilla_growth, safe_pearson


def _trajectory_growth(states):
    states = np.asarray(states, dtype=np.complex128)
    rows = []
    for stop in range(1, states.shape[0] + 1):
        partial = states[:stop]
        equal = trajectory_participation_dimension(partial)
        fs = arc_length_weighted_participation_dimension(partial)
        ceiling = min(partial.shape)
        path = cumulative_fubini_study_length(partial)
        rows.append(
            {
                "depth": stop - 1,
                "d_tp_equal": float(equal.dimension),
                "d_tp_fs": float(fs),
                "d_tp_equal_normalized": float(equal.dimension / ceiling),
                "d_tp_fs_normalized": float(fs / ceiling),
                "trajectory_rank": int(equal.numerical_rank),
                "path_length_fs": float(path[-1]),
            }
        )
    return rows


def _coupled_growth(states, config):
    trajectory = _trajectory_growth(states)
    ancilla = ancilla_growth(states, config.n_qubits, config.n_ancilla)
    if len(trajectory) != len(ancilla):
        raise RuntimeError("trajectory and ancilla diagnostics have different lengths")
    return [{**trow, **{k: v for k, v in arow.items() if k != "depth"}} for trow, arow in zip(trajectory, ancilla)]


def _correlation_fields(growth):
    equal = np.asarray([row["d_tp_equal"] for row in growth], dtype=float)
    fs = np.asarray([row["d_tp_fs"] for row in growth], dtype=float)
    entropy = np.asarray([row["ancilla_entropy_fraction"] for row in growth], dtype=float)
    participation = np.asarray(
        [row["ancilla_participation_dimension"] for row in growth], dtype=float
    )
    mixedness = np.asarray([row["ancilla_mixedness_fraction"] for row in growth], dtype=float)

    result = {
        "corr_dtp_equal_entropy": safe_pearson(equal, entropy),
        "corr_dtp_fs_entropy": safe_pearson(fs, entropy),
        "corr_dtp_equal_ancilla_participation": safe_pearson(equal, participation),
        "corr_dtp_fs_ancilla_participation": safe_pearson(fs, participation),
        "corr_dtp_equal_mixedness": safe_pearson(equal, mixedness),
        "corr_dtp_fs_mixedness": safe_pearson(fs, mixedness),
    }

    if len(growth) >= 3:
        result.update(
            {
                "corr_delta_dtp_equal_delta_entropy": safe_pearson(
                    np.diff(equal), np.diff(entropy)
                ),
                "corr_delta_dtp_fs_delta_entropy": safe_pearson(
                    np.diff(fs), np.diff(entropy)
                ),
                "corr_delta_dtp_equal_delta_ancilla_participation": safe_pearson(
                    np.diff(equal), np.diff(participation)
                ),
                "corr_delta_dtp_fs_delta_ancilla_participation": safe_pearson(
                    np.diff(fs), np.diff(participation)
                ),
            }
        )
    else:
        result.update(
            {
                "corr_delta_dtp_equal_delta_entropy": None,
                "corr_delta_dtp_fs_delta_entropy": None,
                "corr_delta_dtp_equal_delta_ancilla_participation": None,
                "corr_delta_dtp_fs_delta_ancilla_participation": None,
            }
        )
    return result


def evaluate_coupling(config, inputs, weights, *, snapshot_fn=native_shared_qnn_snapshots):
    if not isinstance(config, NativeArchitectureConfig):
        raise TypeError("config must be a NativeArchitectureConfig")
    weights = np.asarray(weights, dtype=float)
    expected = (config.n_layers, config.n_qubits, 3)
    if weights.shape != expected:
        raise ValueError(f"weights must have shape {expected}")
    inputs = np.asarray(inputs, dtype=float).reshape(-1)
    if inputs.size != config.n_features:
        raise ValueError("inputs must contain exactly n_features values")

    states = np.asarray(
        snapshot_fn(
            inputs,
            weights,
            n_qubits=config.n_qubits,
            fm_style=config.fm_style,
            reup_style=config.reup_style,
        ),
        dtype=np.complex128,
    )
    expected_shape = (config.n_layers + 1, 2**config.n_qubits)
    if states.shape != expected_shape:
        raise ValueError(
            f"snapshot function returned shape {states.shape}, expected {expected_shape}"
        )

    growth = _coupled_growth(states, config)
    final = growth[-1]
    coupling = _correlation_fields(growth)
    return states, final, growth, coupling


def run_ancilla_coupling_study(
    *,
    layers=(2, 4, 8),
    n_features=3,
    ancilla_counts=(0, 1, 2, 3),
    feature_maps=("zzfm", "Y"),
    reupload_styles=(None, "X"),
    samples=3,
    seed=2026,
    weight_scale=0.05,
    input_low=-1.0,
    input_high=1.0,
    snapshot_fn=native_shared_qnn_snapshots,
):
    """Compare trajectory reach with instantaneous system--ancilla entanglement.

    The study is intentionally diagnostic.  It uses untrained matched parameter
    draws unless a caller supplies another snapshot function/weight source.  A
    depth-wise correlation is an association between two different diagnostics,
    not evidence that one quantity causes or estimates the other.
    """
    layers, n_features, ancilla_counts, samples, weight_scale = _validate_grid(
        layers, n_features, ancilla_counts, samples, weight_scale
    )
    configs = tuple(
        iter_configs(
            layers=layers,
            n_features=n_features,
            ancilla_counts=ancilla_counts,
            feature_maps=feature_maps,
            reupload_styles=reupload_styles,
        )
    )
    if not configs:
        raise ValueError("architecture grid cannot be empty")

    max_layers = max(config.n_layers for config in configs)
    max_qubits = max(config.n_qubits for config in configs)
    draws = _sample_matched_inputs_and_weights(
        seed=seed,
        samples=samples,
        n_features=n_features,
        max_layers=max_layers,
        max_qubits=max_qubits,
        weight_scale=weight_scale,
        input_low=input_low,
        input_high=input_high,
    )

    rows = []
    growth_rows = []
    for sample, inputs, master_weights in draws:
        for config in configs:
            weights = master_weights[: config.n_layers, : config.n_qubits, :]
            states, final, growth, coupling = evaluate_coupling(
                config, inputs, weights, snapshot_fn=snapshot_fn
            )
            base = {
                "n_layers": config.n_layers,
                "n_features": config.n_features,
                "n_ancilla": config.n_ancilla,
                "n_qubits": config.n_qubits,
                "fm_style": config.fm_style,
                "reup_style": config.reup_style or "none",
                "sample": sample,
                "seed": seed,
            }
            rows.append(
                {
                    **base,
                    "n_snapshots": int(states.shape[0]),
                    "hilbert_dim": int(states.shape[1]),
                    **{f"final_{key}": value for key, value in final.items()},
                    **coupling,
                }
            )
            growth_rows.extend({**base, **entry} for entry in growth)

    return rows, summarize_coupling_rows(rows), growth_rows, summarize_growth_rows(growth_rows)


def _group(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    return groups


def _finite_values(group, metric):
    values = []
    for row in group:
        value = row.get(metric)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def summarize_coupling_rows(rows):
    if not rows:
        return []
    keys = ("n_layers", "n_features", "n_ancilla", "n_qubits", "fm_style", "reup_style")
    metrics = (
        "final_d_tp_equal",
        "final_d_tp_fs",
        "final_ancilla_entropy",
        "final_ancilla_entropy_fraction",
        "final_ancilla_purity",
        "final_ancilla_participation_dimension",
        "final_ancilla_mixedness_fraction",
        "corr_dtp_equal_entropy",
        "corr_dtp_fs_entropy",
        "corr_dtp_equal_ancilla_participation",
        "corr_dtp_fs_ancilla_participation",
        "corr_delta_dtp_equal_delta_entropy",
        "corr_delta_dtp_fs_delta_entropy",
    )
    summary = []
    for key, group in _group(rows, keys).items():
        record = {**dict(zip(keys, key)), "samples": len(group)}
        for metric in metrics:
            values = _finite_values(group, metric)
            record[f"{metric}_n"] = int(values.size)
            record[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
            record[f"{metric}_std"] = float(np.std(values)) if values.size else None
        summary.append(record)
    return sorted(summary, key=lambda row: (row["n_qubits"], row["n_layers"], row["fm_style"], row["reup_style"]))


def summarize_growth_rows(rows):
    if not rows:
        return []
    keys = (
        "n_layers",
        "n_features",
        "n_ancilla",
        "n_qubits",
        "fm_style",
        "reup_style",
        "depth",
    )
    metrics = (
        "d_tp_equal",
        "d_tp_fs",
        "d_tp_equal_normalized",
        "d_tp_fs_normalized",
        "trajectory_rank",
        "path_length_fs",
        "ancilla_entropy",
        "ancilla_entropy_fraction",
        "ancilla_purity",
        "ancilla_participation_dimension",
        "ancilla_participation_normalized",
        "ancilla_mixedness_fraction",
    )
    summary = []
    for key, group in _group(rows, keys).items():
        record = {**dict(zip(keys, key)), "samples": len(group)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=float)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values))
        summary.append(record)
    return sorted(summary, key=lambda row: (row["n_qubits"], row["n_layers"], row["fm_style"], row["reup_style"], row["depth"]))


def _write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir, rows, summary, growth_rows, growth_summary, *, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "ancilla_coupling_raw.csv", rows)
    _write_csv(output_dir / "ancilla_coupling_summary.csv", summary)
    _write_csv(output_dir / "ancilla_coupling_growth_raw.csv", growth_rows)
    _write_csv(output_dir / "ancilla_coupling_growth_summary.csv", growth_summary)
    (output_dir / "ancilla_coupling_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
