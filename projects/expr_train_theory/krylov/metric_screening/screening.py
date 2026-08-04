from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..circuit_ensemble.study import EnsembleConfig, EnsembleRecord, run_ensemble_study
from ..metric_relationships.analysis import _average_ranks, _safe_pearson


_DEFAULT_METRICS = (
    "layer_orbit_fraction",
    "repeated_cycle_fraction",
    "projector_distance",
    "layer_gram_condition",
    "final_mean_single_qubit_entropy",
    "max_mean_single_qubit_entropy",
)


@dataclass(frozen=True)
class ScreeningConfig:
    ensemble: EnsembleConfig
    metrics: tuple[str, ...] = _DEFAULT_METRICS
    variance_tolerance: float = 1e-12
    redundancy_threshold: float = 0.95

    def validate(self) -> None:
        self.ensemble.validate()
        if not self.metrics:
            raise ValueError("at least one metric is required")
        unknown = set(self.metrics) - set(EnsembleRecord.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown record fields: {sorted(unknown)}")
        if self.variance_tolerance < 0:
            raise ValueError("variance_tolerance must be non-negative")
        if not 0 <= self.redundancy_threshold <= 1:
            raise ValueError("redundancy_threshold must lie in [0, 1]")


def screen_records(
    records: Sequence[EnsembleRecord],
    metrics: Sequence[str] = _DEFAULT_METRICS,
    *,
    variance_tolerance: float = 1e-12,
    redundancy_threshold: float = 0.95,
) -> list[dict[str, float | int | str | bool]]:
    if not records:
        raise ValueError("records must not be empty")
    if variance_tolerance < 0:
        raise ValueError("variance_tolerance must be non-negative")
    if not 0 <= redundancy_threshold <= 1:
        raise ValueError("redundancy_threshold must lie in [0, 1]")

    unknown = set(metrics) - set(EnsembleRecord.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown record fields: {sorted(unknown)}")

    rows: list[dict[str, float | int | str | bool]] = []
    values = {name: np.array([float(getattr(record, name)) for record in records]) for name in metrics}
    depths = np.array([record.depth for record in records])
    topologies = np.array([record.topology for record in records], dtype=object)

    for name in metrics:
        x = values[name]
        variance = float(np.var(x))
        unique_values = int(np.unique(np.round(x, decimals=12)).size)
        depth_eta = eta_squared(x, depths)
        topology_eta = eta_squared(x, topologies)

        correlations: list[tuple[str, float]] = []
        for other in metrics:
            if other == name:
                continue
            coefficient = _safe_pearson(_average_ranks(x), _average_ranks(values[other]))
            if np.isfinite(coefficient):
                correlations.append((other, abs(float(coefficient))))
        correlations.sort(key=lambda item: item[1], reverse=True)
        closest_metric, max_abs_spearman = correlations[0] if correlations else ("", float("nan"))

        constant = variance <= variance_tolerance
        highly_redundant = bool(np.isfinite(max_abs_spearman) and max_abs_spearman >= redundancy_threshold)
        rows.append(
            {
                "metric": name,
                "mean": float(np.mean(x)),
                "std": float(np.std(x, ddof=0)),
                "minimum": float(np.min(x)),
                "maximum": float(np.max(x)),
                "unique_values": unique_values,
                "depth_eta_squared": depth_eta,
                "topology_eta_squared": topology_eta,
                "closest_metric": closest_metric,
                "max_abs_spearman": max_abs_spearman,
                "constant_in_sample": constant,
                "highly_redundant_in_sample": highly_redundant,
                "note": _note(constant, highly_redundant, depth_eta, topology_eta),
            }
        )
    return rows


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    labels = np.asarray(groups).reshape(-1)
    if x.size != labels.size or x.size == 0:
        raise ValueError("values and groups must have the same non-zero length")
    total = float(np.sum((x - x.mean()) ** 2))
    if total <= 1e-15:
        return 0.0
    between = 0.0
    for label in np.unique(labels):
        group = x[labels == label]
        between += group.size * float((group.mean() - x.mean()) ** 2)
    return float(between / total)


def run_screening(
    config: ScreeningConfig,
    output_directory: str | Path,
) -> tuple[list[EnsembleRecord], list[dict[str, float | int | str | bool]]]:
    config.validate()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    ensemble_output = output / "ensemble"
    records = run_ensemble_study(config.ensemble, ensemble_output)
    rows = screen_records(
        records,
        config.metrics,
        variance_tolerance=config.variance_tolerance,
        redundancy_threshold=config.redundancy_threshold,
    )
    _write_csv(output / "metric_screening.csv", rows)
    with (output / "metric_screening_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "ensemble": {
                    **asdict(config.ensemble),
                    "depths": list(config.ensemble.depths),
                    "topologies": list(config.ensemble.topologies),
                },
                "metrics": list(config.metrics),
                "variance_tolerance": config.variance_tolerance,
                "redundancy_threshold": config.redundancy_threshold,
                "note": "Exploratory screening only. The flags depend on the sampled circuit family.",
            },
            handle,
            indent=2,
        )
    return records, rows


def _note(constant: bool, redundant: bool, depth_eta: float, topology_eta: float) -> str:
    parts: list[str] = []
    if constant:
        parts.append("no variation in this sample")
    if redundant:
        parts.append("strong rank correlation with another metric")
    if not constant:
        if depth_eta >= 0.5 and topology_eta < 0.2:
            parts.append("mostly tracks depth here")
        elif topology_eta >= 0.5 and depth_eta < 0.2:
            parts.append("mostly separates topologies here")
        elif depth_eta >= 0.2 and topology_eta >= 0.2:
            parts.append("responds to both depth and topology here")
        else:
            parts.append("weak group-level separation in this sample")
    return "; ".join(parts)


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    entries = list(rows)
    if not entries:
        raise ValueError("there are no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
