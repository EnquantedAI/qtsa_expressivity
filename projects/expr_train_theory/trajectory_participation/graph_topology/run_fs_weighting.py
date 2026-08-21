from pathlib import Path

from .fs_weighting import (
    matched_topology_weighting_deltas,
    save_topology_weighting_results,
    summarize_topology_weighting,
    topology_fs_weighting_study,
)


def main():
    rows = topology_fs_weighting_study(
        n_qubits=4,
        n_layers=2,
        parameter_samples=4,
        data_points=4,
    )
    deltas = matched_topology_weighting_deltas(rows)
    summary = summarize_topology_weighting(rows, deltas)

    output_dir = Path(__file__).resolve().parent / "results"
    save_topology_weighting_results(
        output_dir,
        rows,
        deltas,
        summary,
        {
            "purpose": "matched topology comparison of equal-weight and FS-weighted dTP",
            "baseline_topology": "none",
            "interpretation": (
                "sampling diagnostic only; FS weighting changes the snapshot measure and "
                "is not assumed to be a definitive replacement for equal-weight dTP"
            ),
        },
    )

    print("Topology effect under equal and Fubini--Study-weighted dTP")
    print("topology       edges   dTP equal   dTP FS   delta equal   delta FS   FS-equal effect")
    for row in summary:
        print(
            f"{row['topology']:<13} {row['n_edges']:>5d}   "
            f"{row['d_tp_equal_mean']:>9.5f}   {row['d_tp_fs_mean']:>7.5f}   "
            f"{row['delta_d_tp_equal_mean']:>+11.5f}   "
            f"{row['delta_d_tp_fs_mean']:>+8.5f}   "
            f"{row['delta_fs_minus_equal_mean']:>+15.5f}"
        )


if __name__ == "__main__":
    main()
