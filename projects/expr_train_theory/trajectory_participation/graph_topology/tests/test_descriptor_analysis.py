import math
import tempfile
import unittest
from pathlib import Path

from projects.expr_train_theory.trajectory_participation.graph_topology.descriptor_analysis import (
    descriptor_associations,
    matched_metric_deltas,
    save_descriptor_analysis,
    summarize_matched_deltas,
)


DESCRIPTORS = {
    "n_edges": 0,
    "density": 0.0,
    "mean_degree": 0.0,
    "max_degree": 0,
    "algebraic_connectivity": 0.0,
    "diameter": float("inf"),
    "mean_shortest_path": float("inf"),
}

METRICS = {
    "d_tp_mean": 1.0,
    "qfim_rank_mean": 1.0,
    "qfim_trace_mean": 2.0,
    "qntk_rank": 1.0,
    "qntk_effective_rank": 1.0,
    "qntk_trace": 3.0,
}


def row(topology, sample, *, edge_count, shift):
    descriptors = {
        **DESCRIPTORS,
        "n_edges": edge_count,
        "density": edge_count / 3.0,
        "mean_degree": 2.0 * edge_count / 3.0,
        "max_degree": edge_count,
        "algebraic_connectivity": float(edge_count),
        "diameter": float("inf") if edge_count == 0 else float(4 - edge_count),
        "mean_shortest_path": float("inf") if edge_count == 0 else float(4 - edge_count),
    }
    metrics = {name: value + shift for name, value in METRICS.items()}
    return {
        "topology": topology,
        "parameter_sample": sample,
        **descriptors,
        **metrics,
    }


class DescriptorAnalysisTests(unittest.TestCase):
    def test_matched_deltas_use_same_parameter_sample_baseline(self):
        rows = [
            row("none", 0, edge_count=0, shift=0.2),
            row("none", 1, edge_count=0, shift=0.7),
            row("line", 0, edge_count=1, shift=0.5),
            row("line", 1, edge_count=1, shift=1.1),
        ]
        deltas = matched_metric_deltas(rows)
        line = [item for item in deltas if item["topology"] == "line"]
        self.assertAlmostEqual(line[0]["delta_d_tp_mean"], 0.3)
        self.assertAlmostEqual(line[1]["delta_d_tp_mean"], 0.4)

    def test_baseline_deltas_are_zero(self):
        deltas = matched_metric_deltas([row("none", 0, edge_count=0, shift=0.4)])
        self.assertAlmostEqual(deltas[0]["delta_qfim_trace_mean"], 0.0)
        self.assertAlmostEqual(deltas[0]["delta_qntk_effective_rank"], 0.0)

    def test_missing_baseline_is_rejected(self):
        with self.assertRaises(ValueError):
            matched_metric_deltas([row("line", 0, edge_count=1, shift=0.2)])

    def test_summary_and_association_recover_linear_synthetic_trend(self):
        rows = []
        for sample in (0, 1):
            rows.append(row("none", sample, edge_count=0, shift=0.1 * sample))
            for name, edges in (("line", 1), ("ring", 2), ("complete", 3)):
                rows.append(
                    row(
                        name,
                        sample,
                        edge_count=edges,
                        shift=0.1 * sample + float(edges),
                    )
                )
        summary = summarize_matched_deltas(matched_metric_deltas(rows))
        associations = descriptor_associations(
            summary,
            descriptors=("n_edges",),
            metrics=("d_tp_mean",),
        )
        self.assertEqual(len(associations), 1)
        self.assertAlmostEqual(associations[0]["pearson_r"], 1.0)
        self.assertEqual(associations[0]["n_topologies"], 3)

    def test_constant_metric_returns_nan_association(self):
        rows = []
        for name, edges in (("none", 0), ("line", 1), ("ring", 2)):
            rows.append(row(name, 0, edge_count=edges, shift=0.0))
        summary = summarize_matched_deltas(matched_metric_deltas(rows))
        associations = descriptor_associations(
            summary,
            descriptors=("n_edges",),
            metrics=("d_tp_mean",),
        )
        self.assertTrue(math.isnan(associations[0]["pearson_r"]))

    def test_save_writes_all_outputs(self):
        rows = [row("none", 0, edge_count=0, shift=0.0), row("line", 0, edge_count=1, shift=1.0)]
        deltas = matched_metric_deltas(rows)
        summary = summarize_matched_deltas(deltas)
        associations = descriptor_associations(summary, descriptors=("n_edges",), metrics=("d_tp_mean",))
        with tempfile.TemporaryDirectory() as tmp:
            save_descriptor_analysis(tmp, deltas, summary, associations, {"test": True})
            names = {path.name for path in Path(tmp).iterdir()}
        self.assertEqual(
            names,
            {
                "graph_topology_matched_deltas_raw.csv",
                "graph_topology_matched_deltas_summary.csv",
                "graph_topology_descriptor_associations.csv",
                "graph_topology_descriptor_analysis_metadata.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
