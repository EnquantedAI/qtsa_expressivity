"""Controlled QFIM sweep over circuit width and depth.

This is a small diagnostic sweep. It varies only the
number of qubits and variational layers while holding the feature-map family,
parameter initialization and numerical QFIM convention fixed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..core import compute_pure_state_qfim, diagnose_qfim
from ..pennylane_adapter import build_shared_architecture_state_model


@dataclass(frozen=True)
class ArchitectureSweepConfig:
    """Settings for the width-depth sweep."""

    widths: tuple[int, ...] = (2, 3, 4)
    depths: tuple[int, ...] = (1, 2, 3)
    samples_per_architecture: int = 3
    feature_map: str = "zzfm"
    feature_map_repeats: int = 1
    device_name: str = "default.qubit"
    seed: int = 2026
    input_low: float = 0.0
    input_high: float = float(np.pi)
    parameter_low: float = 0.0
    parameter_high: float = float(2.0 * np.pi)
    finite_difference_step: float = 1e-6
    normalization_tolerance: float = 1e-8
    absolute_rank_tolerance: float = 1e-9
    relative_rank_tolerance: float = 1e-7

    def validate(self) -> None:
        if not self.widths or any(int(value) <= 0 for value in self.widths):
            raise ValueError("widths must contain positive integers.")
        if not self.depths or any(int(value) <= 0 for value in self.depths):
            raise ValueError("depths must contain positive integers.")
        if self.samples_per_architecture <= 0:
            raise ValueError("samples_per_architecture must be positive.")
        if self.input_low >= self.input_high:
            raise ValueError("input_low must be smaller than input_high.")
        if self.parameter_low >= self.parameter_high:
            raise ValueError("parameter_low must be smaller than parameter_high.")
        if self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive.")


RAW_FIELDNAMES = (
    "width",
    "depth",
    "sample_index",
    "sample_seed",
    "feature_map",
    "parameter_count",
    "numerical_rank",
    "relative_rank",
    "trace",
    "minimum_eigenvalue",
    "positive_condition_number",
    "symmetry_error",
    "input_vector",
    "eigenvalues",
)

SUMMARY_FIELDNAMES = (
    "width",
    "depth",
    "feature_map",
    "sample_count",
    "parameter_count",
    "mean_rank",
    "std_rank",
    "mean_relative_rank",
    "std_relative_rank",
    "mean_trace",
    "std_trace",
    "mean_minimum_eigenvalue",
    "mean_positive_condition_number",
)


def _json_array(values: Sequence[float]) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def aggregate_architecture_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate rows by width, depth and feature map."""

    groups: dict[tuple[int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["width"]), int(row["depth"]), str(row["feature_map"]))
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for (width, depth, feature_map), group in sorted(groups.items()):
        ranks = np.asarray([float(row["numerical_rank"]) for row in group], dtype=float)
        relative_ranks = np.asarray([float(row["relative_rank"]) for row in group], dtype=float)
        traces = np.asarray([float(row["trace"]) for row in group], dtype=float)
        minimum_eigenvalues = np.asarray(
            [float(row["minimum_eigenvalue"]) for row in group], dtype=float
        )
        condition_numbers = np.asarray(
            [float(row["positive_condition_number"]) for row in group], dtype=float
        )
        finite_condition_numbers = condition_numbers[np.isfinite(condition_numbers)]

        summaries.append(
            {
                "width": width,
                "depth": depth,
                "feature_map": feature_map,
                "sample_count": len(group),
                "parameter_count": int(group[0]["parameter_count"]),
                "mean_rank": float(np.mean(ranks)),
                "std_rank": float(np.std(ranks, ddof=0)),
                "mean_relative_rank": float(np.mean(relative_ranks)),
                "std_relative_rank": float(np.std(relative_ranks, ddof=0)),
                "mean_trace": float(np.mean(traces)),
                "std_trace": float(np.std(traces, ddof=0)),
                "mean_minimum_eigenvalue": float(np.mean(minimum_eigenvalues)),
                "mean_positive_condition_number": (
                    float(np.mean(finite_condition_numbers))
                    if finite_condition_numbers.size
                    else float("inf")
                ),
            }
        )

    return summaries


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_architecture_sweep(
    config: ArchitectureSweepConfig,
    *,
    output_directory: str | Path,
) -> tuple[Path, Path, Path]:
    """Run the sweep and save raw results, a summary and metadata."""

    config.validate()
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []

    for width in config.widths:
        for depth in config.depths:
            for sample_index in range(config.samples_per_architecture):
                sample_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
                rng = np.random.default_rng(sample_seed)
                inputs = rng.uniform(config.input_low, config.input_high, size=int(width))

                model = build_shared_architecture_state_model(
                    inputs,
                    n_layers=int(depth),
                    feature_map=config.feature_map,
                    device_name=config.device_name,
                    feature_map_repeats=config.feature_map_repeats,
                )
                parameters = rng.uniform(
                    config.parameter_low,
                    config.parameter_high,
                    size=model.parameter_count,
                )

                matrix = compute_pure_state_qfim(
                    model.state_function,
                    parameters,
                    step=config.finite_difference_step,
                    normalization_tolerance=config.normalization_tolerance,
                )
                diagnostics = diagnose_qfim(
                    matrix,
                    absolute_rank_tolerance=config.absolute_rank_tolerance,
                    relative_rank_tolerance=config.relative_rank_tolerance,
                )

                rows.append(
                    {
                        "width": int(width),
                        "depth": int(depth),
                        "sample_index": sample_index,
                        "sample_seed": sample_seed,
                        "feature_map": config.feature_map,
                        "parameter_count": model.parameter_count,
                        "numerical_rank": diagnostics.numerical_rank,
                        "relative_rank": diagnostics.relative_rank,
                        "trace": diagnostics.trace,
                        "minimum_eigenvalue": diagnostics.minimum_eigenvalue,
                        "positive_condition_number": diagnostics.positive_condition_number,
                        "symmetry_error": diagnostics.symmetry_error,
                        "input_vector": _json_array(inputs),
                        "eigenvalues": _json_array(diagnostics.eigenvalues),
                    }
                )

    summaries = aggregate_architecture_rows(rows)
    raw_path = output_dir / "qfim_architecture_sweep_raw.csv"
    summary_path = output_dir / "qfim_architecture_sweep_summary.csv"
    metadata_path = output_dir / "qfim_architecture_sweep_metadata.json"

    _write_csv(raw_path, RAW_FIELDNAMES, rows)
    _write_csv(summary_path, SUMMARY_FIELDNAMES, summaries)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "small_qfim_width_depth_sweep",
        "qfim_convention": (
            "4 Re(<d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi>)"
        ),
        "derivatives": "central_finite_differences",
        "config": asdict(config),
        "raw_rows": len(rows),
        "summary_rows": len(summaries),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return raw_path, summary_path, metadata_path
