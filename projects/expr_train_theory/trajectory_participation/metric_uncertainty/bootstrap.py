from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import numpy as np


@dataclass(frozen=True)
class BootstrapEstimate:
    estimate: float
    lower: float
    upper: float
    n: int
    bootstrap_samples: int


def _percentile_interval(values, confidence):
    values = np.asarray(values, dtype=float)
    alpha = 1.0 - float(confidence)
    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lower, upper


def bootstrap_mean(values, *, bootstrap_samples=2000, confidence=0.95, seed=2026):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("values must contain at least one finite value")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(bootstrap_samples, values.size))
    boot = np.mean(values[indices], axis=1)
    lower, upper = _percentile_interval(boot, confidence)
    return BootstrapEstimate(
        estimate=float(np.mean(values)),
        lower=lower,
        upper=upper,
        n=int(values.size),
        bootstrap_samples=int(bootstrap_samples),
    )


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def bootstrap_correlation(
    x,
    y,
    *,
    method="pearson",
    bootstrap_samples=2000,
    confidence=0.95,
    seed=2026,
):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        raise ValueError("at least three paired finite values are required")
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")

    def corr(a, b):
        if method == "spearman":
            a = _rankdata(a)
            b = _rankdata(b)
        return _pearson(a, b)

    estimate = corr(x, y)
    rng = np.random.default_rng(seed)
    boot = []
    attempts = 0
    max_attempts = bootstrap_samples * 10
    while len(boot) < bootstrap_samples and attempts < max_attempts:
        idx = rng.integers(0, x.size, size=x.size)
        value = corr(x[idx], y[idx])
        if np.isfinite(value):
            boot.append(value)
        attempts += 1
    if not boot:
        raise ValueError("correlation is undefined for the supplied sample")

    lower, upper = _percentile_interval(boot, confidence)
    return BootstrapEstimate(
        estimate=float(estimate),
        lower=lower,
        upper=upper,
        n=int(x.size),
        bootstrap_samples=len(boot),
    )


def compare_metric_pairs(rows, pairs, *, bootstrap_samples=2000, confidence=0.95, seed=2026):
    rows = list(rows)
    output = []
    for index, (left, right) in enumerate(pairs):
        x = [row[left] for row in rows]
        y = [row[right] for row in rows]
        for method in ("pearson", "spearman"):
            try:
                estimate = bootstrap_correlation(
                    x,
                    y,
                    method=method,
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                    seed=seed + index,
                )
                output.append(
                    {
                        "left": left,
                        "right": right,
                        "method": method,
                        **asdict(estimate),
                    }
                )
            except ValueError:
                output.append(
                    {
                        "left": left,
                        "right": right,
                        "method": method,
                        "estimate": float("nan"),
                        "lower": float("nan"),
                        "upper": float("nan"),
                        "n": len(rows),
                        "bootstrap_samples": 0,
                    }
                )
    return output


def save_report(output_dir, rows, metadata):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    csv_path = output / "metric_uncertainty.csv"
    json_path = output / "metric_uncertainty_metadata.json"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return csv_path, json_path
