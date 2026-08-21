from pathlib import Path

from .robust_summary import build_robust_topology_summary, save_robust_summary
from .uncertainty_sweep import (
    bootstrap_matched_delta_uncertainty,
    repeat_seeds,
    topology_scaling_multiseed,
)


def main():
    qubits = (3, 4)
    layers = (1, 2, 3)
    base_seed = 2026
    repeats = 8
    seed_stride = 10_000
    bootstrap_samples = 2000
    confidence = 0.95

    repeat_rows = topology_scaling_multiseed(
        qubits=qubits,
        layers=layers,
        parameter_samples=2,
        data_points=3,
        base_seed=base_seed,
        repeats=repeats,
        seed_stride=seed_stride,
    )
    uncertainty = bootstrap_matched_delta_uncertainty(
        repeat_rows,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=base_seed + 777,
    )
    configurations, summary = build_robust_topology_summary(uncertainty)

    output_dir = Path(__file__).resolve().parent / "results"
    save_robust_summary(
        output_dir,
        configurations,
        summary,
        {
            "purpose": "classify robust matched topology effects from multi-seed uncertainty",
            "qubits": list(qubits),
            "layers": list(layers),
            "baseline_topology": "none",
            "repeat_seeds": list(
                repeat_seeds(base_seed=base_seed, repeats=repeats, stride=seed_stride)
            ),
            "parameter_samples_per_repeat_configuration": 2,
            "data_points_per_repeat_configuration": 3,
            "bootstrap_samples": bootstrap_samples,
            "confidence": confidence,
            "configuration_classification": (
                "stable_positive/negative only when the bootstrap interval excludes zero"
            ),
            "aggregate_classification": (
                "stable_positive/negative only when every sampled width/depth configuration "
                "is resolved in that same direction; otherwise unresolved"
            ),
            "interpretation": (
                "the labels summarize this finite sampled grid and are not universal topology laws"
            ),
        },
    )

    print("Robust matched topology effects")
    print(
        "topology       metric                              pos neg unresolved "
        "resolved  direction   robust"
    )
    for row in summary:
        print(
            f"{row['topology']:<13} {row['metric']:<35} "
            f"{row['stable_positive']:>3d} {row['stable_negative']:>3d} "
            f"{row['unresolved']:>10d} {row['resolved_fraction']:>8.3f}  "
            f"{row['resolved_direction']:<10} {row['robust_classification']}"
        )


if __name__ == "__main__":
    main()
