from pathlib import Path

import numpy as np

from .cross_metric import topology_cross_metric_study
from .descriptor_analysis import (
    descriptor_associations,
    matched_metric_deltas,
    save_descriptor_analysis,
    summarize_matched_deltas,
)


def _format_r(value):
    return "n/a" if not np.isfinite(value) else f"{value:+.3f}"


def main():
    rows = topology_cross_metric_study(
        n_qubits=4,
        n_layers=2,
        parameter_samples=3,
        data_points=4,
    )
    deltas = matched_metric_deltas(rows)
    summary = summarize_matched_deltas(deltas)
    associations = descriptor_associations(summary)

    output_dir = Path(__file__).resolve().parent / "results"
    save_descriptor_analysis(
        output_dir,
        deltas,
        summary,
        associations,
        {
            "purpose": "matched graph-descriptor associations with dTP/QFIM/QNTK shifts",
            "baseline_topology": "none",
            "association": "topology-level Pearson correlation",
            "interpretation": (
                "exploratory diagnostic only; the small topology set is not a predictive model "
                "and no monotonic graph-metric relation is assumed"
            ),
        },
    )

    print("Matched mean shifts relative to topology='none'")
    print("topology       edges   delta dTP   delta QFIM rank   delta QNTK eff.rank")
    for row in summary:
        print(
            f"{row['topology']:<13} {row['n_edges']:>5d}   "
            f"{row['delta_d_tp_mean_mean']:>+9.5f}   "
            f"{row['delta_qfim_rank_mean_mean']:>+15.3f}   "
            f"{row['delta_qntk_effective_rank_mean']:>+19.5f}"
        )

    print("\nSelected descriptor associations (non-baseline topologies only)")
    selected = {
        ("density", "d_tp_mean"),
        ("algebraic_connectivity", "d_tp_mean"),
        ("density", "qfim_rank_mean"),
        ("algebraic_connectivity", "qntk_effective_rank"),
    }
    for row in associations:
        if (row["descriptor"], row["metric"]) in selected:
            print(
                f"{row['descriptor']:<24} vs {row['metric']:<20} "
                f"r={_format_r(row['pearson_r'])}  n={row['n_topologies']}"
            )


if __name__ == "__main__":
    main()
