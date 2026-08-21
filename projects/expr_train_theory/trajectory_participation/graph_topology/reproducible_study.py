from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .robust_summary import build_robust_topology_summary, save_robust_summary
from .uncertainty_sweep import (
    UNCERTAINTY_METRICS,
    bootstrap_matched_delta_uncertainty,
    repeat_seeds,
    save_uncertainty_results,
    summarize_ci_stability,
    topology_scaling_multiseed,
)


REPRODUCIBLE_STUDY_SCHEMA_VERSION = 1
REPRODUCIBLE_STUDY_NAME = "topology_robust_v1"
DEFAULT_TOPOLOGIES = ("none", "line", "ring", "star", "complete")


@dataclass(frozen=True)
class TopologyStudyPreset:
    """Complete parameter set for a reproducible topology study."""

    name: str = REPRODUCIBLE_STUDY_NAME
    qubits: tuple[int, ...] = (3, 4)
    layers: tuple[int, ...] = (1, 2, 3)
    topologies: tuple[str, ...] = DEFAULT_TOPOLOGIES
    parameter_samples: int = 2
    data_points: int = 3
    base_seed: int = 2026
    repeats: int = 8
    seed_stride: int = 10_000
    step: float = 1e-6
    bootstrap_samples: int = 2000
    confidence: float = 0.95
    bootstrap_seed_offset: int = 777
    baseline_topology: str = "none"

    def validate(self):
        if not self.name:
            raise ValueError("preset name must be non-empty")
        if not self.qubits or any(value < 1 for value in self.qubits):
            raise ValueError("qubits must contain positive integers")
        if not self.layers or any(value < 1 for value in self.layers):
            raise ValueError("layers must contain positive integers")
        if not self.topologies:
            raise ValueError("topologies must be non-empty")
        if self.baseline_topology not in self.topologies:
            raise ValueError("baseline_topology must be included in topologies")
        if self.parameter_samples < 1 or self.data_points < 1:
            raise ValueError("parameter_samples and data_points must be positive")
        if self.repeats < 1 or self.seed_stride < 1:
            raise ValueError("repeats and seed_stride must be positive")
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be between zero and one")
        return self


DEFAULT_REPRODUCIBLE_PRESET = TopologyStudyPreset()


def preset_manifest(preset=DEFAULT_REPRODUCIBLE_PRESET):
    """Return the exact configuration recorded with an official study run."""
    preset.validate()
    manifest = asdict(preset)
    manifest.update(
        {
            "schema_version": REPRODUCIBLE_STUDY_SCHEMA_VERSION,
            "repeat_seeds": list(
                repeat_seeds(
                    base_seed=preset.base_seed,
                    repeats=preset.repeats,
                    stride=preset.seed_stride,
                )
            ),
            "bootstrap_seed": preset.base_seed + preset.bootstrap_seed_offset,
            "uncertainty_metrics": list(UNCERTAINTY_METRICS),
            "runner": (
                "python -m projects.expr_train_theory.trajectory_participation."
                "graph_topology.run_reproducible_study"
            ),
            "bootstrap_unit": "independent repeat seed after matched topology comparison",
            "interpretation": (
                "finite sampled width/depth grid; robust labels are diagnostics and are not "
                "universal or monotonic topology laws"
            ),
        }
    )
    # JSON should use arrays rather than Python tuple notation.
    manifest["qubits"] = list(preset.qubits)
    manifest["layers"] = list(preset.layers)
    manifest["topologies"] = list(preset.topologies)
    return manifest


def run_reproducible_topology_study(preset=DEFAULT_REPRODUCIBLE_PRESET):
    """Run the complete matched topology pipeline for one frozen preset."""
    preset.validate()
    repeat_rows = topology_scaling_multiseed(
        qubits=preset.qubits,
        layers=preset.layers,
        topologies=preset.topologies,
        parameter_samples=preset.parameter_samples,
        data_points=preset.data_points,
        base_seed=preset.base_seed,
        repeats=preset.repeats,
        seed_stride=preset.seed_stride,
        step=preset.step,
        baseline_topology=preset.baseline_topology,
    )
    uncertainty_rows = bootstrap_matched_delta_uncertainty(
        repeat_rows,
        baseline_topology=preset.baseline_topology,
        bootstrap_samples=preset.bootstrap_samples,
        confidence=preset.confidence,
        seed=preset.base_seed + preset.bootstrap_seed_offset,
    )
    ci_stability_rows = summarize_ci_stability(
        uncertainty_rows,
        baseline_topology=preset.baseline_topology,
    )
    robust_configuration_rows, robust_summary_rows = build_robust_topology_summary(
        uncertainty_rows,
        baseline_topology=preset.baseline_topology,
    )
    return {
        "manifest": preset_manifest(preset),
        "repeat_rows": repeat_rows,
        "uncertainty_rows": uncertainty_rows,
        "ci_stability_rows": ci_stability_rows,
        "robust_configuration_rows": robust_configuration_rows,
        "robust_summary_rows": robust_summary_rows,
    }


def save_reproducible_topology_study(output_dir, result):
    """Save every table needed to reproduce and inspect the official study."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(result["manifest"])

    save_uncertainty_results(
        output_dir,
        result["repeat_rows"],
        result["uncertainty_rows"],
        result["ci_stability_rows"],
        manifest,
    )
    save_robust_summary(
        output_dir,
        result["robust_configuration_rows"],
        result["robust_summary_rows"],
        manifest,
    )
    with (output_dir / "graph_topology_reproducible_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    return tuple(sorted(path.name for path in output_dir.iterdir() if path.is_file()))
