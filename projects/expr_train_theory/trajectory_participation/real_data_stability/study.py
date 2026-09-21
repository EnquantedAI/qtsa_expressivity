from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from ..real_data_bridge.data import PROJECT_DATASET_NAMES
from ..real_data_bridge.study import run_real_data_study
from ..shared_qnn import native_shared_qnn_snapshots


DEFAULT_STABILITY_METRICS = (
    "final_d_tp_equal_normalized",
    "final_d_tp_fs_normalized",
    "final_d_tp_equal",
    "final_d_tp_fs",
    "final_path_length_fs",
    "final_ancilla_entropy_fraction",
    "final_ancilla_participation_dimension",
)


def _validate_repeat_seeds(repeat_seeds):
    seeds = tuple(int(seed) for seed in repeat_seeds)
    if not seeds:
        raise ValueError("repeat_seeds cannot be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("repeat_seeds must be unique")
    return seeds


def _finite(value):
    return value is not None and np.isfinite(value)


def _metric_delta(row, baseline, metric):
    lhs = row.get(metric)
    rhs = baseline.get(metric)
    if not _finite(lhs) or not _finite(rhs):
        return None
    return float(lhs) - float(rhs)


def matched_effect_rows(
    rows,
    *,
    metrics=DEFAULT_STABILITY_METRICS,
    baseline_ancilla=0,
    baseline_depth=None,
):
    """Build per-window, per-weight-draw matched architecture deltas.

    Two effect families are kept separate:
    * ``ancilla`` compares ancilla counts at fixed depth/encoding;
    * ``depth`` compares depths at fixed ancilla count/encoding.

    The function never averages before subtraction, so every delta remains matched
    on dataset window and random parameter draw.
    """
    if not rows:
        return []
    metrics = tuple(metrics)
    depths = sorted({int(row["n_layers"]) for row in rows})
    if baseline_depth is None:
        baseline_depth = depths[0]
    baseline_depth = int(baseline_depth)
    baseline_ancilla = int(baseline_ancilla)
    if baseline_depth not in depths:
        raise ValueError("baseline_depth is not present in rows")
    if baseline_ancilla not in {int(row["n_ancilla"]) for row in rows}:
        raise ValueError("baseline_ancilla is not present in rows")

    exact = {}
    for row in rows:
        key = (
            row["dataset"],
            row["split"],
            int(row["window_index"]),
            int(row.get("weight_seed", row["seed"])),
            int(row["weight_sample"]),
            int(row["n_layers"]),
            int(row["n_ancilla"]),
            row["fm_style"],
            row["reup_style"],
        )
        if key in exact:
            raise ValueError(f"duplicate matched observation for key {key}")
        exact[key] = row

    output = []
    for key, row in exact.items():
        dataset, split, window_index, weight_seed, weight_sample, n_layers, n_ancilla, fm, reup = key

        if n_ancilla != baseline_ancilla:
            bkey = (
                dataset,
                split,
                window_index,
                weight_seed,
                weight_sample,
                n_layers,
                baseline_ancilla,
                fm,
                reup,
            )
            baseline = exact.get(bkey)
            if baseline is not None:
                record = {
                    "effect_type": "ancilla",
                    "dataset": dataset,
                    "split": split,
                    "window_index": window_index,
                    "weight_seed": weight_seed,
                    "weight_sample": weight_sample,
                    "n_features": int(row["n_features"]),
                    "fm_style": fm,
                    "reup_style": reup,
                    "n_layers": n_layers,
                    "baseline_n_ancilla": baseline_ancilla,
                    "comparison_n_ancilla": n_ancilla,
                    "n_ancilla": None,
                    "baseline_n_layers": None,
                    "comparison_n_layers": None,
                }
                record.update({f"delta_{metric}": _metric_delta(row, baseline, metric) for metric in metrics})
                output.append(record)

        if n_layers != baseline_depth:
            bkey = (
                dataset,
                split,
                window_index,
                weight_seed,
                weight_sample,
                baseline_depth,
                n_ancilla,
                fm,
                reup,
            )
            baseline = exact.get(bkey)
            if baseline is not None:
                record = {
                    "effect_type": "depth",
                    "dataset": dataset,
                    "split": split,
                    "window_index": window_index,
                    "weight_seed": weight_seed,
                    "weight_sample": weight_sample,
                    "n_features": int(row["n_features"]),
                    "fm_style": fm,
                    "reup_style": reup,
                    "n_layers": None,
                    "baseline_n_ancilla": None,
                    "comparison_n_ancilla": None,
                    "n_ancilla": n_ancilla,
                    "baseline_n_layers": baseline_depth,
                    "comparison_n_layers": n_layers,
                }
                record.update({f"delta_{metric}": _metric_delta(row, baseline, metric) for metric in metrics})
                output.append(record)

    order = lambda r: (
        r["dataset"],
        r["effect_type"],
        r["fm_style"],
        r["reup_style"],
        -1 if r["n_layers"] is None else r["n_layers"],
        -1 if r["n_ancilla"] is None else r["n_ancilla"],
        -1 if r["comparison_n_layers"] is None else r["comparison_n_layers"],
        -1 if r["comparison_n_ancilla"] is None else r["comparison_n_ancilla"],
        r["window_index"],
        r["weight_seed"],
        r["weight_sample"],
    )
    return sorted(output, key=order)


def _summary_group_key(row):
    return (
        row["effect_type"],
        row["dataset"],
        row["split"],
        row["n_features"],
        row["fm_style"],
        row["reup_style"],
        row["n_layers"],
        row["baseline_n_ancilla"],
        row["comparison_n_ancilla"],
        row["n_ancilla"],
        row["baseline_n_layers"],
        row["comparison_n_layers"],
    )


def _crossed_bootstrap_means(rows, metric, *, samples, rng):
    field = f"delta_{metric}"
    valid = [row for row in rows if _finite(row.get(field))]
    if not valid:
        return np.asarray([], dtype=float)

    windows = sorted({int(row["window_index"]) for row in valid})
    draws = sorted({(int(row["weight_seed"]), int(row["weight_sample"])) for row in valid})
    cells = {}
    for row in valid:
        cells[(int(row["window_index"]), (int(row["weight_seed"]), int(row["weight_sample"])))] = float(row[field])

    result = []
    for _ in range(int(samples)):
        sampled_windows = rng.choice(windows, size=len(windows), replace=True)
        draw_indices = rng.integers(0, len(draws), size=len(draws))
        sampled_draws = [draws[index] for index in draw_indices]
        values = []
        for window in sampled_windows:
            for draw in sampled_draws:
                value = cells.get((int(window), draw))
                if value is not None:
                    values.append(value)
        if values:
            result.append(float(np.mean(values)))
    return np.asarray(result, dtype=float)


def _mean_std_by(rows, field, group_fields):
    groups = {}
    for row in rows:
        value = row.get(field)
        if not _finite(value):
            continue
        key = tuple(row[name] for name in group_fields)
        groups.setdefault(key, []).append(float(value))
    means = np.asarray([np.mean(values) for values in groups.values()], dtype=float)
    return float(np.std(means)) if means.size else None


def summarize_effect_stability(
    effect_rows,
    *,
    metrics=DEFAULT_STABILITY_METRICS,
    bootstrap_samples=1000,
    confidence=0.95,
    bootstrap_seed=314159,
    zero_tolerance=1e-12,
):
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    groups = {}
    for row in effect_rows:
        groups.setdefault(_summary_group_key(row), []).append(row)

    rng = np.random.default_rng(bootstrap_seed)
    alpha = (1.0 - confidence) / 2.0
    summary = []
    for key, group in groups.items():
        common = {
            name: value
            for name, value in zip(
                (
                    "effect_type",
                    "dataset",
                    "split",
                    "n_features",
                    "fm_style",
                    "reup_style",
                    "n_layers",
                    "baseline_n_ancilla",
                    "comparison_n_ancilla",
                    "n_ancilla",
                    "baseline_n_layers",
                    "comparison_n_layers",
                ),
                key,
            )
        }
        for metric in metrics:
            field = f"delta_{metric}"
            values = np.asarray(
                [float(row[field]) for row in group if _finite(row.get(field))],
                dtype=float,
            )
            if not values.size:
                continue
            boot = _crossed_bootstrap_means(
                group,
                metric,
                samples=bootstrap_samples,
                rng=rng,
            )
            if boot.size:
                ci_low, ci_high = np.quantile(boot, [alpha, 1.0 - alpha])
                ci_low, ci_high = float(ci_low), float(ci_high)
            else:
                ci_low = ci_high = None

            positive = int(np.sum(values > zero_tolerance))
            negative = int(np.sum(values < -zero_tolerance))
            zero = int(values.size - positive - negative)
            if ci_low is not None and ci_low > zero_tolerance:
                label = "stable_positive"
            elif ci_high is not None and ci_high < -zero_tolerance:
                label = "stable_negative"
            else:
                label = "unresolved"

            summary.append(
                {
                    **common,
                    "metric": metric,
                    "observations": int(values.size),
                    "unique_windows": len({row["window_index"] for row in group}),
                    "unique_weight_seeds": len({row["weight_seed"] for row in group}),
                    "unique_weight_draws": len(
                        {(row["weight_seed"], row["weight_sample"]) for row in group}
                    ),
                    "mean_delta": float(np.mean(values)),
                    "std_delta": float(np.std(values)),
                    "window_mean_std": _mean_std_by(group, field, ("window_index",)),
                    "weight_draw_mean_std": _mean_std_by(
                        group, field, ("weight_seed", "weight_sample")
                    ),
                    "positive_fraction": positive / values.size,
                    "negative_fraction": negative / values.size,
                    "zero_fraction": zero / values.size,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "confidence": float(confidence),
                    "classification": label,
                }
            )

    return sorted(
        summary,
        key=lambda row: (
            row["dataset"],
            row["effect_type"],
            row["fm_style"],
            row["reup_style"],
            str(row["metric"]),
            -1 if row["n_layers"] is None else row["n_layers"],
            -1 if row["n_ancilla"] is None else row["n_ancilla"],
            -1 if row["comparison_n_layers"] is None else row["comparison_n_layers"],
            -1 if row["comparison_n_ancilla"] is None else row["comparison_n_ancilla"],
        ),
    )


def run_real_data_stability(
    *,
    dataset_names=PROJECT_DATASET_NAMES,
    split="test",
    n_windows=8,
    window_selection="even",
    window_seed=1907,
    layers=(2, 4, 8),
    ancilla_counts=(0, 1, 2),
    feature_maps=("zzfm",),
    reupload_styles=("X",),
    repeat_seeds=(2026, 2027, 2028, 2029, 2030),
    weight_samples_per_seed=1,
    weight_scale=0.05,
    baseline_ancilla=0,
    baseline_depth=None,
    metrics=DEFAULT_STABILITY_METRICS,
    bootstrap_samples=1000,
    confidence=0.95,
    bootstrap_seed=314159,
    repo_root=None,
    snapshot_fn=native_shared_qnn_snapshots,
):
    """Measure architecture-effect stability across real inputs and weight seeds.

    The held-out windows are fixed across repeat seeds.  Each architecture effect
    is subtracted within the same dataset window and random parameter draw before
    any aggregation.  Confidence intervals use a crossed bootstrap over windows
    and parameter draws, so neither source of variability is silently treated as
    fixed.
    """
    repeat_seeds = _validate_repeat_seeds(repeat_seeds)
    all_rows = []
    selection_reference = None
    per_seed_metadata = []

    for weight_seed in repeat_seeds:
        rows, _summary, _growth, _growth_summary, metadata = run_real_data_study(
            dataset_names=dataset_names,
            split=split,
            n_windows=n_windows,
            window_selection=window_selection,
            layers=layers,
            ancilla_counts=ancilla_counts,
            feature_maps=feature_maps,
            reupload_styles=reupload_styles,
            weight_samples=weight_samples_per_seed,
            seed=weight_seed,
            window_seed=window_seed,
            weight_scale=weight_scale,
            repo_root=repo_root,
            snapshot_fn=snapshot_fn,
        )
        selected = metadata["selected_window_indices"]
        if selection_reference is None:
            selection_reference = selected
        elif selected != selection_reference:
            raise RuntimeError("real-data window selection changed across repeat seeds")
        for row in rows:
            row = dict(row)
            row["weight_seed"] = int(weight_seed)
            all_rows.append(row)
        per_seed_metadata.append(
            {
                "weight_seed": int(weight_seed),
                "weight_samples": int(weight_samples_per_seed),
            }
        )

    effects = matched_effect_rows(
        all_rows,
        metrics=metrics,
        baseline_ancilla=baseline_ancilla,
        baseline_depth=baseline_depth,
    )
    stability = summarize_effect_stability(
        effects,
        metrics=metrics,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
    )
    if baseline_depth is None:
        baseline_depth = min(int(value) for value in layers)

    metadata = {
        "datasets": list(dataset_names),
        "split": split,
        "n_windows_per_dataset": int(n_windows),
        "window_selection": window_selection,
        "window_seed": int(window_seed),
        "selected_window_indices": selection_reference,
        "layers": [int(value) for value in layers],
        "ancilla_counts": [int(value) for value in ancilla_counts],
        "feature_maps": list(feature_maps),
        "reupload_styles": [value if value is not None else "none" for value in reupload_styles],
        "repeat_seeds": list(repeat_seeds),
        "weight_samples_per_seed": int(weight_samples_per_seed),
        "weight_scale": float(weight_scale),
        "baseline_ancilla": int(baseline_ancilla),
        "baseline_depth": int(baseline_depth),
        "metrics": list(metrics),
        "bootstrap_method": "crossed resampling of held-out windows and weight draws",
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "confidence": float(confidence),
        "weights_trained": False,
        "targets_used_in_metrics": False,
        "per_seed_runs": per_seed_metadata,
    }
    return all_rows, effects, stability, metadata


def _write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir, raw_rows, effect_rows, stability_summary, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "real_data_stability_raw.csv", raw_rows)
    _write_csv(output_dir / "real_data_stability_effects.csv", effect_rows)
    _write_csv(output_dir / "real_data_stability_summary.csv", stability_summary)
    (output_dir / "real_data_stability_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
