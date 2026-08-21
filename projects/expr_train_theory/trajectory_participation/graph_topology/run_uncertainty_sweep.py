from pathlib import Path

from .uncertainty_sweep import (
    bootstrap_matched_delta_uncertainty,
    repeat_seeds,
    save_uncertainty_results,
    summarize_ci_stability,
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
    stability = summarize_ci_stability(uncertainty)

    output_dir = Path(__file__).resolve().parent / "results"
    save_uncertainty_results(
        output_dir,
        repeat_rows,
        uncertainty,
        stability,
        {
            "purpose": "estimate repeat-to-repeat uncertainty of matched topology shifts",
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
            "bootstrap_unit": "independent repeat seed after matched topology comparison",
            "interpretation": (
                "percentile intervals are finite-sample diagnostics; an interval excluding zero "
                "does not by itself establish a universal or monotonic topology law"
            ),
        },
    )

    print("Multi-seed matched topology uncertainty")
    print(
        "qubits layers topology       metric                              "
        "mean delta        CI                         repeat sign"
    )
    for row in uncertainty:
        if row["topology"] == "none":
            continue
        print(
            f"{row['n_qubits']:>6d} {row['n_layers']:>6d} {row['topology']:<13} "
            f"{row['metric']:<35} {row['mean_delta']:>+10.5f} "
            f"[{row['ci_lower']:>+9.5f}, {row['ci_upper']:>+9.5f}] "
            f"{row['ci_relation']:<13}"
        )

    print("\nCI stability across width/depth")
    print("topology       metric                              resolved  sign       consistency")
    for row in stability:
        consistency = row["resolved_sign_consistency"]
        consistency_text = "n/a" if consistency != consistency else f"{consistency:.3f}"
        print(
            f"{row['topology']:<13} {row['metric']:<35} "
            f"{row['resolved_fraction']:>8.3f}  "
            f"{row['dominant_resolved_sign']:<9} {consistency_text:>11}"
        )


if __name__ == "__main__":
    main()
