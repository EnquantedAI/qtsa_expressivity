import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from ..arc_length_weighting.weights import (
    arc_length_snapshot_weights,
    arc_length_weighted_participation_dimension,
    cumulative_fubini_study_length,
)
from ..architecture_sweep.sweep import SweepConfig
from ..core import trajectory_participation_dimension
from ..shared_qnn import shared_qnn_snapshots


def _sample_input(rng, input_size):
    if input_size < 1:
        raise ValueError("input_size must be positive")
    return rng.uniform(-np.pi, np.pi, size=input_size)


def _sample_weights(rng, config):
    return rng.uniform(-np.pi, np.pi, size=(config.n_layers, config.n_qubits, 3))


def compare_config(
    config,
    *,
    samples=5,
    input_size=3,
    seed=2026,
    snapshot_fn=shared_qnn_snapshots,
):
    """Evaluate equal and arc-length-weighted dTP on matched trajectories."""
    if samples < 1:
        raise ValueError("samples must be positive")

    rng = np.random.default_rng(seed)
    rows = []
    for sample in range(samples):
        inputs = _sample_input(rng, input_size)
        weights = _sample_weights(rng, config)
        states = snapshot_fn(
            inputs,
            weights,
            n_qubits=config.n_qubits,
            fm_style=config.fm_style,
            reup_style=config.reup_style,
        )

        equal = trajectory_participation_dimension(states)
        arc = arc_length_weighted_participation_dimension(states)
        arc_weights = arc_length_snapshot_weights(states)
        cumulative = cumulative_fubini_study_length(states)
        max_dimension = min(states.shape)

        rows.append(
            {
                "n_layers": config.n_layers,
                "n_qubits": config.n_qubits,
                "fm_style": config.fm_style,
                "reup_style": config.reup_style or "none",
                "sample": sample,
                "seed": seed,
                "input_size": input_size,
                "n_snapshots": int(states.shape[0]),
                "hilbert_dim": int(states.shape[1]),
                "path_length_fs": float(cumulative[-1]),
                "d_tp_equal": float(equal.dimension),
                "d_tp_arc": float(arc),
                "d_tp_equal_normalized": float(equal.dimension / max_dimension),
                "d_tp_arc_normalized": float(arc / max_dimension),
                "arc_minus_equal": float(arc - equal.dimension),
                "arc_over_equal": float(arc / equal.dimension),
                "arc_weight_min": float(np.min(arc_weights)),
                "arc_weight_max": float(np.max(arc_weights)),
                "numerical_rank": int(equal.numerical_rank),
            }
        )
    return rows


def summarize_rows(rows):
    if not rows:
        return []

    keys = ("n_layers", "n_qubits", "fm_style", "reup_style")
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)

    summary = []
    for key, group in groups.items():
        def values(name):
            return np.asarray([row[name] for row in group], dtype=float)

        equal = values("d_tp_equal")
        arc = values("d_tp_arc")
        delta = values("arc_minus_equal")
        path = values("path_length_fs")
        summary.append(
            {
                **dict(zip(keys, key)),
                "samples": len(group),
                "d_tp_equal_mean": float(np.mean(equal)),
                "d_tp_equal_std": float(np.std(equal)),
                "d_tp_arc_mean": float(np.mean(arc)),
                "d_tp_arc_std": float(np.std(arc)),
                "arc_minus_equal_mean": float(np.mean(delta)),
                "arc_minus_equal_abs_mean": float(np.mean(np.abs(delta))),
                "path_length_fs_mean": float(np.mean(path)),
            }
        )

    return sorted(
        summary,
        key=lambda row: (
            row["n_qubits"],
            row["n_layers"],
            row["fm_style"],
            row["reup_style"],
        ),
    )


def run_comparison_sweep(
    *,
    layers=(1, 2, 3, 4),
    qubits=(1, 2, 3),
    feature_maps=("zzfm", "iqp", "Y"),
    reupload_styles=(None, "Y"),
    samples=5,
    input_size=3,
    seed=2026,
    snapshot_fn=shared_qnn_snapshots,
):
    rows = []
    index = 0
    for n_layers, n_qubits, fm_style, reup_style in product(
        layers, qubits, feature_maps, reupload_styles
    ):
        config = SweepConfig(
            n_layers=int(n_layers),
            n_qubits=int(n_qubits),
            fm_style=str(fm_style),
            reup_style=reup_style,
        )
        rows.extend(
            compare_config(
                config,
                samples=samples,
                input_size=input_size,
                seed=seed + index,
                snapshot_fn=snapshot_fn,
            )
        )
        index += 1
    return rows, summarize_rows(rows)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir, rows, summary, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "arc_length_architecture_raw.csv", rows)
    _write_csv(output_dir / "arc_length_architecture_summary.csv", summary)
    with (output_dir / "arc_length_architecture_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
