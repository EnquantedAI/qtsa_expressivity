from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..circuit_ensemble.study import EnsembleConfig, EnsembleRecord, run_ensemble_study


_DEFAULT_METRICS = (
    "layer_orbit_fraction",
    "repeated_cycle_fraction",
    "projector_distance",
    "layer_gram_condition",
    "final_mean_single_qubit_entropy",
    "max_mean_single_qubit_entropy",
)


@dataclass(frozen=True)
class CorrelationConfig:
    ensemble: EnsembleConfig
    metrics: tuple[str, ...] = _DEFAULT_METRICS

    def validate(self) -> None:
        self.ensemble.validate()
        if len(self.metrics) < 2:
            raise ValueError("at least two metrics are required")
        unknown = set(self.metrics) - set(EnsembleRecord.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown record fields: {sorted(unknown)}")


def analyse_records(
    records: Sequence[EnsembleRecord],
    metrics: Sequence[str] = _DEFAULT_METRICS,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | int | str]]]:
    if not records:
        raise ValueError("records must not be empty")
    matrix = _values(records, metrics)
    global_rows = correlation_matrix(matrix, metrics)

    grouped_rows: list[dict[str, float | int | str]] = []
    groups: dict[tuple[int, str], list[EnsembleRecord]] = {}
    for record in records:
        groups.setdefault((record.depth, record.topology), []).append(record)

    for (depth, topology), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        values = _values(group, metrics)
        for row in correlation_matrix(values, metrics):
            grouped_rows.append(
                {
                    "depth": depth,
                    "topology": topology,
                    "samples": len(group),
                    **row,
                }
            )
    return global_rows, grouped_rows


def correlation_matrix(
    values: np.ndarray,
    metric_names: Sequence[str],
) -> list[dict[str, float | str]]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(metric_names):
        raise ValueError("values must have one column per metric")
    rows: list[dict[str, float | str]] = []
    for i, first in enumerate(metric_names):
        for j in range(i + 1, len(metric_names)):
            second = metric_names[j]
            x = data[:, i]
            y = data[:, j]
            rows.append(
                {
                    "metric_a": first,
                    "metric_b": second,
                    "pearson": _safe_pearson(x, y),
                    "spearman": _safe_pearson(_average_ranks(x), _average_ranks(y)),
                }
            )
    return rows


def run_correlation_study(
    config: CorrelationConfig,
    output_directory: str | Path,
) -> tuple[list[EnsembleRecord], list[dict[str, float | str]], list[dict[str, float | int | str]]]:
    config.validate()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    ensemble_output = output / "ensemble"
    records = run_ensemble_study(config.ensemble, ensemble_output)
    global_rows, grouped_rows = analyse_records(records, config.metrics)

    _write_csv(output / "metric_correlations_global.csv", global_rows)
    _write_csv(output / "metric_correlations_grouped.csv", grouped_rows)
    with (output / "metric_correlations_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "ensemble": {
                    **asdict(config.ensemble),
                    "depths": list(config.ensemble.depths),
                    "topologies": list(config.ensemble.topologies),
                },
                "metrics": list(config.metrics),
                "note": "Exploratory correlation check. Correlation is not evidence that two metrics are equivalent.",
            },
            handle,
            indent=2,
        )
    return records, global_rows, grouped_rows


def _values(records: Sequence[EnsembleRecord], metrics: Sequence[str]) -> np.ndarray:
    unknown = set(metrics) - set(EnsembleRecord.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown record fields: {sorted(unknown)}")
    return np.array([[float(getattr(record, name)) for name in metrics] for record in records])


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    first = np.asarray(x, dtype=float)
    second = np.asarray(y, dtype=float)
    if first.size != second.size or first.size < 2:
        return float("nan")
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-15:
        return float("nan")
    return float(np.dot(first, second) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        average = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    entries = list(rows)
    if not entries:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
