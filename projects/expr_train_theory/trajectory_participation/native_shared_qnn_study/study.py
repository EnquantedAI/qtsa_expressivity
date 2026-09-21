from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import csv
import json

import numpy as np

from ..arc_length_weighting.weights import (
    arc_length_weighted_participation_dimension,
    cumulative_fubini_study_length,
)
from ..core import trajectory_participation_dimension
from ..shared_qnn import native_shared_qnn_snapshots


@dataclass(frozen=True)
class NativeArchitectureConfig:
    """One shared-QNN architecture in the native PennyLane study."""

    n_layers: int
    n_features: int
    n_ancilla: int
    fm_style: str
    reup_style: str | None = None

    def __post_init__(self):
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")
        if self.n_features < 1:
            raise ValueError("n_features must be positive")
        if self.n_ancilla < 0:
            raise ValueError("n_ancilla cannot be negative")
        if self.fm_style not in {"X", "Y", "Z", "zzfm", "iqp"}:
            raise ValueError("unsupported feature map")
        if self.reup_style not in {None, "X", "Y", "Z"}:
            raise ValueError("unsupported reuploading style")

    @property
    def n_qubits(self):
        return self.n_features + self.n_ancilla


def iter_configs(
    *,
    layers,
    n_features,
    ancilla_counts,
    feature_maps,
    reupload_styles,
):
    for n_layers, n_ancilla, fm_style, reup_style in product(
        layers, ancilla_counts, feature_maps, reupload_styles
    ):
        yield NativeArchitectureConfig(
            n_layers=int(n_layers),
            n_features=int(n_features),
            n_ancilla=int(n_ancilla),
            fm_style=str(fm_style),
            reup_style=reup_style,
        )


def _validate_grid(layers, n_features, ancilla_counts, samples, weight_scale):
    layers = tuple(int(value) for value in layers)
    ancilla_counts = tuple(int(value) for value in ancilla_counts)
    if not layers or any(value < 1 for value in layers):
        raise ValueError("layers must contain positive integers")
    if int(n_features) < 1:
        raise ValueError("n_features must be positive")
    if not ancilla_counts or any(value < 0 for value in ancilla_counts):
        raise ValueError("ancilla_counts must contain non-negative integers")
    if int(samples) < 1:
        raise ValueError("samples must be positive")
    if float(weight_scale) < 0.0:
        raise ValueError("weight_scale cannot be negative")
    return layers, int(n_features), ancilla_counts, int(samples), float(weight_scale)


def _sample_matched_inputs_and_weights(
    *,
    seed,
    samples,
    n_features,
    max_layers,
    max_qubits,
    weight_scale,
    input_low,
    input_high,
):
    """Draw one nested parameter tensor per sample for the full architecture grid.

    Every architecture in a given sample reuses the same input vector and slices
    the same master tensor to its requested depth and width.  This keeps the
    common parameters matched across depth, ancilla count, feature map and
    reuploading comparisons without pretending that architectures of different
    sizes have identical parameter spaces.
    """
    if input_high <= input_low:
        raise ValueError("input_high must be greater than input_low")

    rng = np.random.default_rng(seed)
    draws = []
    for sample in range(samples):
        inputs = rng.uniform(input_low, input_high, size=n_features)
        master_weights = rng.uniform(
            -np.pi,
            np.pi,
            size=(max_layers, max_qubits, 3),
        ) * weight_scale
        draws.append((sample, inputs, master_weights))
    return draws


def _growth_rows(states):
    states = np.asarray(states, dtype=np.complex128)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("states must be a non-empty 2D trajectory")

    output = []
    for stop in range(1, states.shape[0] + 1):
        partial = states[:stop]
        equal = trajectory_participation_dimension(partial)
        fs = arc_length_weighted_participation_dimension(partial)
        ceiling = min(partial.shape)
        path = cumulative_fubini_study_length(partial)
        output.append(
            {
                "depth": stop - 1,
                "n_snapshots": stop,
                "d_tp_equal": float(equal.dimension),
                "d_tp_fs": float(fs),
                "d_tp_equal_normalized": float(equal.dimension / ceiling),
                "d_tp_fs_normalized": float(fs / ceiling),
                "trajectory_rank": int(equal.numerical_rank),
                "path_length_fs": float(path[-1]),
            }
        )
    return output


def evaluate_trajectory(config, inputs, weights, *, snapshot_fn=native_shared_qnn_snapshots):
    """Evaluate one native shared-QNN trajectory and its depth-resolved metrics."""
    weights = np.asarray(weights, dtype=float)
    expected = (config.n_layers, config.n_qubits, 3)
    if weights.shape != expected:
        raise ValueError(f"weights must have shape {expected}")

    inputs = np.asarray(inputs, dtype=float).reshape(-1)
    if inputs.size != config.n_features:
        raise ValueError("inputs must contain exactly n_features values")

    states = snapshot_fn(
        inputs,
        weights,
        n_qubits=config.n_qubits,
        fm_style=config.fm_style,
        reup_style=config.reup_style,
    )
    states = np.asarray(states, dtype=np.complex128)
    expected_shape = (config.n_layers + 1, 2**config.n_qubits)
    if states.shape != expected_shape:
        raise ValueError(
            f"snapshot function returned shape {states.shape}, expected {expected_shape}"
        )

    growth = _growth_rows(states)
    final = growth[-1]
    return states, final, growth


def run_native_architecture_study(
    *,
    layers=(1, 2, 4),
    n_features=3,
    ancilla_counts=(0, 1, 2),
    feature_maps=("zzfm", "Y"),
    reupload_styles=(None, "X"),
    samples=3,
    seed=2026,
    weight_scale=0.05,
    input_low=-1.0,
    input_high=1.0,
    snapshot_fn=native_shared_qnn_snapshots,
):
    """Run a native shared-QNN trajectory study on a matched architecture grid.

    This is an architecture diagnostic, not a trained-model performance study.
    Inputs and random untrained parameters are deliberately controlled so the
    metric pipeline can be compared across width/depth/encoding choices before
    moving to trained weights and held-out project data.
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
            states, final, growth = evaluate_trajectory(
                config,
                inputs,
                weights,
                snapshot_fn=snapshot_fn,
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
                }
            )
            growth_rows.extend({**base, **entry} for entry in growth)

    return rows, summarize_final_rows(rows), growth_rows, summarize_growth_rows(growth_rows)


def _group(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    return groups


def summarize_final_rows(rows):
    if not rows:
        return []
    keys = ("n_layers", "n_features", "n_ancilla", "n_qubits", "fm_style", "reup_style")
    metrics = (
        "final_d_tp_equal",
        "final_d_tp_fs",
        "final_d_tp_equal_normalized",
        "final_d_tp_fs_normalized",
        "final_trajectory_rank",
        "final_path_length_fs",
    )
    summary = []
    for key, group in _group(rows, keys).items():
        record = {**dict(zip(keys, key)), "samples": len(group)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=float)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values))
        summary.append(record)
    return sorted(
        summary,
        key=lambda row: (
            row["n_qubits"],
            row["n_layers"],
            row["fm_style"],
            row["reup_style"],
        ),
    )


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
    )
    summary = []
    for key, group in _group(rows, keys).items():
        record = {**dict(zip(keys, key)), "samples": len(group)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=float)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values))
        summary.append(record)
    return sorted(
        summary,
        key=lambda row: (
            row["n_qubits"],
            row["n_layers"],
            row["fm_style"],
            row["reup_style"],
            row["depth"],
        ),
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


def save_results(
    output_dir,
    rows,
    summary,
    growth_rows,
    growth_summary,
    *,
    metadata,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "native_shared_qnn_raw.csv", rows)
    _write_csv(output_dir / "native_shared_qnn_summary.csv", summary)
    _write_csv(output_dir / "native_shared_qnn_growth_raw.csv", growth_rows)
    _write_csv(output_dir / "native_shared_qnn_growth_summary.csv", growth_summary)
    (output_dir / "native_shared_qnn_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def config_as_dict(config):
    return asdict(config)
