import csv
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

from ..core import trajectory_participation_dimension
from ..shared_qnn import shared_qnn_snapshots


@dataclass(frozen=True)
class SweepConfig:
    n_layers: int
    n_qubits: int
    fm_style: str
    reup_style: str | None


def iter_configs(layers, qubits, feature_maps, reupload_styles):
    for n_layers, n_qubits, fm_style, reup_style in product(
        layers, qubits, feature_maps, reupload_styles
    ):
        yield SweepConfig(
            n_layers=int(n_layers),
            n_qubits=int(n_qubits),
            fm_style=str(fm_style),
            reup_style=reup_style,
        )


def _sample_input(rng, input_size):
    if input_size < 1:
        raise ValueError("input_size must be positive")
    return rng.uniform(-np.pi, np.pi, size=input_size)


def _sample_weights(rng, config):
    return rng.uniform(
        -np.pi,
        np.pi,
        size=(config.n_layers, config.n_qubits, 3),
    )


def evaluate_config(
    config,
    *,
    samples=5,
    input_size=3,
    seed=2026,
    snapshot_fn=shared_qnn_snapshots,
):
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
        result = trajectory_participation_dimension(states)
        max_dimension = min(states.shape[0], states.shape[1])

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
                "d_tp": result.dimension,
                "d_tp_normalized": result.dimension / max_dimension,
                "numerical_rank": result.numerical_rank,
                "entropy": result.entropy,
                "largest_weight": float(np.max(result.spectrum)),
            }
        )

    return rows


def summarize_rows(rows):
    if not rows:
        return []

    groups = {}
    keys = ("n_layers", "n_qubits", "fm_style", "reup_style")
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)

    summary = []
    for key, group in groups.items():
        d_tp = np.array([row["d_tp"] for row in group], dtype=float)
        d_tp_norm = np.array([row["d_tp_normalized"] for row in group], dtype=float)
        rank = np.array([row["numerical_rank"] for row in group], dtype=float)
        entropy = np.array([row["entropy"] for row in group], dtype=float)

        summary.append(
            {
                **dict(zip(keys, key)),
                "samples": len(group),
                "d_tp_mean": float(np.mean(d_tp)),
                "d_tp_std": float(np.std(d_tp)),
                "d_tp_normalized_mean": float(np.mean(d_tp_norm)),
                "rank_mean": float(np.mean(rank)),
                "entropy_mean": float(np.mean(entropy)),
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


def run_sweep(
    *,
    layers=(1, 2, 3, 4),
    qubits=(1, 2, 3, 4),
    feature_maps=("zzfm", "iqp", "Y"),
    reupload_styles=(None, "Y"),
    samples=5,
    input_size=3,
    seed=2026,
    snapshot_fn=shared_qnn_snapshots,
):
    rows = []
    for index, config in enumerate(iter_configs(layers, qubits, feature_maps, reupload_styles)):
        rows.extend(
            evaluate_config(
                config,
                samples=samples,
                input_size=input_size,
                seed=seed + index,
                snapshot_fn=snapshot_fn,
            )
        )
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


def save_sweep(output_dir, rows, summary, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "trajectory_sweep_raw.csv", rows)
    _write_csv(output_dir / "trajectory_sweep_summary.csv", summary)
    with (output_dir / "trajectory_sweep_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def config_as_dict(config):
    return asdict(config)
