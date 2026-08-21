import csv
import json
from pathlib import Path

import numpy as np

from ..core import trajectory_participation_dimension
from ..snapshots import trajectory_snapshots


def rank_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays of the same shape")
    if x.size < 2:
        return float("nan")

    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        out = np.empty(values.size, dtype=float)
        i = 0
        while i < values.size:
            j = i + 1
            while j < values.size and values[order[j]] == values[order[i]]:
                j += 1
            out[order[i:j]] = 0.5 * (i + j - 1) + 1.0
            i = j
        return out

    rx = ranks(x)
    ry = ranks(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def monotonicity_summary(x, y, *, tolerance=1e-10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays of the same shape")
    if x.size < 2:
        raise ValueError("at least two points are required")

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    unique_x = np.unique(x)
    means = np.array([np.mean(y[x == value]) for value in unique_x])
    diffs = np.diff(means)

    return {
        "spearman": rank_correlation(x, y),
        "positive_steps": int(np.count_nonzero(diffs > tolerance)),
        "negative_steps": int(np.count_nonzero(diffs < -tolerance)),
        "flat_steps": int(np.count_nonzero(np.abs(diffs) <= tolerance)),
        "first_mean": float(means[0]),
        "last_mean": float(means[-1]),
        "delta": float(means[-1] - means[0]),
    }


def _sample_case(rng, *, n_layers, n_qubits, reupload_axis, entangle):
    features = rng.uniform(-np.pi, np.pi, size=n_qubits)
    parameters = rng.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 3))
    states = trajectory_snapshots(
        features,
        parameters,
        n_qubits=n_qubits,
        reupload_axis=reupload_axis,
        entangle=entangle,
    )
    result = trajectory_participation_dimension(states)
    max_dim = min(states.shape)
    return {
        "d_tp": result.dimension,
        "d_tp_normalized": result.dimension / max_dim,
        "rank": result.numerical_rank,
        "entropy": result.entropy,
        "largest_weight": float(np.max(result.spectrum)),
    }


def run_reference_study(
    *,
    layers=(1, 2, 3, 4, 5),
    qubits=(1, 2, 3),
    samples=20,
    seed=2026,
):
    if samples < 1:
        raise ValueError("samples must be positive")

    rng = np.random.default_rng(seed)
    rows = []
    for n_qubits in qubits:
        for n_layers in layers:
            for sample in range(samples):
                # Use the same random seed stream for the two reupload variants.
                local_seed = int(rng.integers(0, 2**32 - 1))
                for reupload_axis in (None, "Y"):
                    local_rng = np.random.default_rng(local_seed)
                    values = _sample_case(
                        local_rng,
                        n_layers=int(n_layers),
                        n_qubits=int(n_qubits),
                        reupload_axis=reupload_axis,
                        entangle=n_qubits > 1,
                    )
                    rows.append(
                        {
                            "n_layers": int(n_layers),
                            "n_qubits": int(n_qubits),
                            "sample": int(sample),
                            "reupload": reupload_axis or "none",
                            **values,
                        }
                    )

    summaries = []
    for n_qubits in qubits:
        for reupload in ("none", "Y"):
            subset = [r for r in rows if r["n_qubits"] == n_qubits and r["reupload"] == reupload]
            for metric in ("d_tp", "d_tp_normalized", "rank", "entropy"):
                result = monotonicity_summary(
                    [r["n_layers"] for r in subset],
                    [r[metric] for r in subset],
                )
                summaries.append(
                    {
                        "scan": "depth",
                        "n_qubits": int(n_qubits),
                        "reupload": reupload,
                        "metric": metric,
                        **result,
                    }
                )

    paired = []
    keys = sorted({(r["n_qubits"], r["n_layers"], r["sample"]) for r in rows})
    lookup = {(r["n_qubits"], r["n_layers"], r["sample"], r["reupload"]): r for r in rows}
    for n_qubits, n_layers, sample in keys:
        base = lookup[(n_qubits, n_layers, sample, "none")]
        reup = lookup[(n_qubits, n_layers, sample, "Y")]
        paired.append(
            {
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "sample": sample,
                "d_tp_delta_reupload": reup["d_tp"] - base["d_tp"],
                "entropy_delta_reupload": reup["entropy"] - base["entropy"],
            }
        )

    return rows, summaries, paired


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir, rows, summaries, paired, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "directional_raw.csv", rows)
    _write_csv(output_dir / "directional_depth_summary.csv", summaries)
    _write_csv(output_dir / "directional_reupload_pairs.csv", paired)
    with (output_dir / "directional_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
