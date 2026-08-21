from pathlib import Path

from .comprehensive import (
    matched_comprehensive_deltas,
    save_comprehensive_results,
    summarize_comprehensive,
    topology_comprehensive_study,
)


def main():
    rows = topology_comprehensive_study(
        n_qubits=4,
        n_layers=2,
        parameter_samples=3,
        data_points=4,
    )
    deltas = matched_comprehensive_deltas(rows)
    summary = summarize_comprehensive(rows, deltas)

    output_dir = Path(__file__).resolve().parent / "results"
    save_comprehensive_results(
        output_dir,
        rows,
        deltas,
        summary,
        {
            "purpose": "single matched topology experiment collecting the main trajectory and trainability diagnostics",
            "baseline_topology": "none",
            "qntk_output": "final-state expectation value <Z_0>",
            "fs_weighting": "Fubini-Study arc-length trapezoidal weights",
            "interpretation": (
                "empirical diagnostic; topology descriptors and metric shifts are not assumed "
                "to satisfy a monotonic or predictive relation"
            ),
        },
    )

    print("Comprehensive matched topology summary")
    print(
        "topology       edges density lambda2   dTP eq   dTP FS   FS path   "
        "QFIM rk   QNTK rk   QNTK eff"
    )
    for row in summary:
        print(
            f"{row['topology']:<13} {row['n_edges']:>5d} "
            f"{row['density']:>7.3f} {row['algebraic_connectivity']:>7.3f} "
            f"{row['d_tp_equal_mean']:>8.5f} {row['d_tp_fs_mean']:>8.5f} "
            f"{row['path_length_fs_mean']:>9.5f} {row['qfim_rank_mean']:>9.3f} "
            f"{row['qntk_rank']:>9.3f} {row['qntk_effective_rank']:>10.5f}"
        )

    print("\nMatched mean shifts relative to topology='none'")
    print(
        "topology       delta dTP eq   delta dTP FS   delta path   "
        "delta QFIM rk   delta QNTK eff"
    )
    for row in summary:
        print(
            f"{row['topology']:<13} "
            f"{row['delta_d_tp_equal_mean']:>+12.5f} "
            f"{row['delta_d_tp_fs_mean']:>+14.5f} "
            f"{row['delta_path_length_fs_mean']:>+11.5f} "
            f"{row['delta_qfim_rank_mean']:>+14.3f} "
            f"{row['delta_qntk_effective_rank']:>+15.5f}"
        )


if __name__ == "__main__":
    main()
