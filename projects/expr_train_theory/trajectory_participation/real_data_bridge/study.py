from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from ..ancilla_coupling.study import evaluate_coupling
from ..native_shared_qnn_study.study import NativeArchitectureConfig, iter_configs
from ..shared_qnn import native_shared_qnn_snapshots
from .data import PROJECT_DATASET_NAMES, load_project_dataset, select_windows


def _validate_study_grid(layers, ancilla_counts, n_windows, weight_samples, weight_scale):
    layers = tuple(int(value) for value in layers)
    ancilla_counts = tuple(int(value) for value in ancilla_counts)
    n_windows = int(n_windows)
    weight_samples = int(weight_samples)
    weight_scale = float(weight_scale)
    if not layers or any(value < 1 for value in layers):
        raise ValueError("layers must contain positive integers")
    if not ancilla_counts or any(value < 0 for value in ancilla_counts):
        raise ValueError("ancilla_counts must contain non-negative integers")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    if weight_samples < 1:
        raise ValueError("weight_samples must be positive")
    if weight_scale < 0.0:
        raise ValueError("weight_scale cannot be negative")
    return layers, ancilla_counts, n_windows, weight_samples, weight_scale


def _master_weight_draws(seed, weight_samples, max_layers, max_qubits, weight_scale):
    rng = np.random.default_rng(seed)
    return [
        rng.uniform(-np.pi, np.pi, size=(max_layers, max_qubits, 3)) * weight_scale
        for _ in range(weight_samples)
    ]


def _group(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    return groups


def _finite_values(group, metric):
    values = [row.get(metric) for row in group]
    return np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )


def summarize_real_data_rows(rows):
    if not rows:
        return []
    keys = (
        "dataset",
        "split",
        "n_layers",
        "n_features",
        "n_ancilla",
        "n_qubits",
        "fm_style",
        "reup_style",
    )
    metrics = (
        "target",
        "final_d_tp_equal",
        "final_d_tp_fs",
        "final_d_tp_equal_normalized",
        "final_d_tp_fs_normalized",
        "final_trajectory_rank",
        "final_path_length_fs",
        "final_ancilla_entropy",
        "final_ancilla_entropy_fraction",
        "final_ancilla_purity",
        "final_ancilla_participation_dimension",
        "final_ancilla_mixedness_fraction",
    )
    output = []
    for key, group in _group(rows, keys).items():
        record = {**dict(zip(keys, key)), "observations": len(group)}
        record["unique_windows"] = len({row["window_index"] for row in group})
        record["weight_samples"] = len({row["weight_sample"] for row in group})
        for metric in metrics:
            values = _finite_values(group, metric)
            record[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
            record[f"{metric}_std"] = float(np.std(values)) if values.size else None
        output.append(record)
    return sorted(
        output,
        key=lambda row: (
            row["dataset"],
            row["n_qubits"],
            row["n_layers"],
            row["fm_style"],
            row["reup_style"],
        ),
    )


def summarize_real_data_growth(rows):
    if not rows:
        return []
    keys = (
        "dataset",
        "split",
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
        "ancilla_mixedness_fraction",
    )
    output = []
    for key, group in _group(rows, keys).items():
        record = {**dict(zip(keys, key)), "observations": len(group)}
        for metric in metrics:
            values = _finite_values(group, metric)
            record[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
            record[f"{metric}_std"] = float(np.std(values)) if values.size else None
        output.append(record)
    return sorted(
        output,
        key=lambda row: (
            row["dataset"],
            row["n_qubits"],
            row["n_layers"],
            row["fm_style"],
            row["reup_style"],
            row["depth"],
        ),
    )


def run_real_data_study(
    *,
    dataset_names=PROJECT_DATASET_NAMES,
    split="test",
    n_windows=8,
    window_selection="even",
    layers=(2, 4, 8),
    ancilla_counts=(0, 1, 2),
    feature_maps=("zzfm",),
    reupload_styles=("X",),
    weight_samples=1,
    seed=2026,
    window_seed=None,
    weight_scale=0.05,
    repo_root=None,
    snapshot_fn=native_shared_qnn_snapshots,
):
    """Evaluate trajectory and ancilla diagnostics on the team's real data windows.

    Inputs come from the already-preprocessed project TensorDatasets.  Targets are
    recorded for provenance but are not used in the state-space metrics.  Weights
    are untrained unless a future caller supplies a different weight source, so
    this study characterises data/architecture response rather than forecasting
    performance.
    """
    layers, ancilla_counts, n_windows, weight_samples, weight_scale = _validate_study_grid(
        layers, ancilla_counts, n_windows, weight_samples, weight_scale
    )
    datasets = [load_project_dataset(name, split, repo_root=repo_root) for name in dataset_names]
    if not datasets:
        raise ValueError("dataset_names cannot be empty")

    n_features = datasets[0].n_features
    for dataset in datasets:
        if dataset.n_features != n_features:
            raise ValueError("all datasets must use the same input-window width")
        if n_windows > dataset.n_windows:
            raise ValueError(
                f"requested {n_windows} windows from {dataset.name}, but only {dataset.n_windows} are available"
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
    master_weights = _master_weight_draws(
        seed, weight_samples, max_layers, max_qubits, weight_scale
    )

    rows = []
    growth_rows = []
    selected_indices = {}
    for dataset_number, dataset in enumerate(datasets):
        selected = select_windows(
            dataset,
            n_windows,
            mode=window_selection,
            seed=(seed + dataset_number) if window_seed is None else (int(window_seed) + dataset_number),
        )
        selected_indices[dataset.name] = [item["window_index"] for item in selected]

        for window in selected:
            for weight_sample, master in enumerate(master_weights):
                for config in configs:
                    weights = master[: config.n_layers, : config.n_qubits, :]
                    states, final, growth, coupling = evaluate_coupling(
                        config,
                        window["inputs"],
                        weights,
                        snapshot_fn=snapshot_fn,
                    )
                    base = {
                        "dataset": dataset.name,
                        "split": dataset.split,
                        "window_index": window["window_index"],
                        "target": window["target"],
                        "weight_sample": weight_sample,
                        "seed": int(seed),
                        "n_layers": config.n_layers,
                        "n_features": config.n_features,
                        "n_ancilla": config.n_ancilla,
                        "n_qubits": config.n_qubits,
                        "fm_style": config.fm_style,
                        "reup_style": config.reup_style or "none",
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

    metadata_root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else datasets[0].source_path.resolve().parents[4]
    )
    metadata = {
        "datasets": [dataset.name for dataset in datasets],
        "split": split,
        "n_features": n_features,
        "n_windows_per_dataset": n_windows,
        "window_selection": window_selection,
        "selected_window_indices": selected_indices,
        "layers": list(layers),
        "ancilla_counts": list(ancilla_counts),
        "feature_maps": list(feature_maps),
        "reupload_styles": [value if value is not None else "none" for value in reupload_styles],
        "weight_samples": weight_samples,
        "seed": int(seed),
        "window_seed": None if window_seed is None else int(window_seed),
        "weight_scale": weight_scale,
        "weights_trained": False,
        "targets_used_in_metrics": False,
        "dataset_sources": {
            dataset.name: str(dataset.source_path.resolve().relative_to(metadata_root))
            for dataset in datasets
        },
    }
    return (
        rows,
        summarize_real_data_rows(rows),
        growth_rows,
        summarize_real_data_growth(growth_rows),
        metadata,
    )


def _write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir, rows, summary, growth_rows, growth_summary, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "real_data_raw.csv", rows)
    _write_csv(output_dir / "real_data_summary.csv", summary)
    _write_csv(output_dir / "real_data_growth_raw.csv", growth_rows)
    _write_csv(output_dir / "real_data_growth_summary.csv", growth_summary)
    (output_dir / "real_data_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
