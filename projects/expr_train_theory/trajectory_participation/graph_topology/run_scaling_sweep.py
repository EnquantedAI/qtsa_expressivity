from pathlib import Path

from .scaling_sweep import (
    save_scaling_results,
    summarize_topology_stability,
    topology_scaling_sweep,
)


def main():
    qubits = (3, 4)
    layers = (1, 2, 3)
    rows = topology_scaling_sweep(
        qubits=qubits,
        layers=layers,
        parameter_samples=2,
        data_points=3,
    )
    stability = summarize_topology_stability(rows)

    output_dir = Path(__file__).resolve().parent / "results"
    save_scaling_results(
        output_dir,
        rows,
        stability,
        {
            "purpose": "check whether matched topology effects persist across circuit width and depth",
            "qubits": list(qubits),
            "layers": list(layers),
            "baseline_topology": "none",
            "parameter_samples_per_configuration": 2,
            "data_points_per_configuration": 3,
            "d_tp_normalization": "divide by min(n_layers + 1, 2**n_qubits)",
            "interpretation": (
                "descriptive stability diagnostic; sign consistency is not a significance test "
                "and does not establish a monotonic topology law"
            ),
        },
    )

    print("Matched topology scaling sweep")
    print(
        "qubits layers topology       dTPnorm eq  dTPnorm FS   "
        "delta eq   delta FS   delta QFIM rel   delta QNTK eff"
    )
    for row in rows:
        print(
            f"{row['n_qubits']:>6d} {row['n_layers']:>6d} {row['topology']:<13} "
            f"{row['d_tp_equal_normalized']:>10.5f} "
            f"{row['d_tp_fs_normalized']:>10.5f} "
            f"{row['delta_d_tp_equal_normalized']:>+10.5f} "
            f"{row['delta_d_tp_fs_normalized']:>+10.5f} "
            f"{row['delta_qfim_relative_rank_mean']:>+16.5f} "
            f"{row['delta_qntk_effective_rank']:>+16.5f}"
        )

    print("\nCross-configuration sign stability")
    print("topology       metric                              mean delta   sign      consistency")
    for row in stability:
        consistency = row["nonzero_sign_consistency"]
        consistency_text = "n/a" if consistency != consistency else f"{consistency:.3f}"
        print(
            f"{row['topology']:<13} {row['metric']:<35} "
            f"{row['mean_delta']:>+10.5f}   {row['dominant_sign']:<8}  {consistency_text:>11}"
        )


if __name__ == "__main__":
    main()
