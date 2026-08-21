import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation.graph_topology.metrics import (
    graph_metrics,
    topology_edges,
)
from projects.expr_train_theory.trajectory_participation.graph_topology.study import (
    summarize,
    topology_snapshots,
    topology_study,
)


class GraphTopologyTests(unittest.TestCase):
    def test_line_graph_metrics(self):
        g = graph_metrics(4, topology_edges(4, "line"))
        self.assertEqual(g.n_edges, 3)
        self.assertAlmostEqual(g.density, 0.5)
        self.assertEqual(g.connected_components, 1)
        self.assertAlmostEqual(g.diameter, 3.0)

    def test_complete_graph_metrics(self):
        g = graph_metrics(4, topology_edges(4, "complete"))
        self.assertEqual(g.n_edges, 6)
        self.assertAlmostEqual(g.density, 1.0)
        self.assertAlmostEqual(g.diameter, 1.0)
        self.assertAlmostEqual(g.algebraic_connectivity, 4.0)

    def test_disconnected_graph_has_zero_algebraic_connectivity(self):
        g = graph_metrics(4, ())
        self.assertEqual(g.connected_components, 4)
        self.assertEqual(g.algebraic_connectivity, 0.0)
        self.assertTrue(np.isinf(g.diameter))

    def test_ring_edges_are_not_duplicated_for_two_nodes(self):
        self.assertEqual(topology_edges(2, "ring"), ((0, 1),))

    def test_snapshot_count_and_norm(self):
        parameters = np.zeros((3, 3, 3))
        states = topology_snapshots([0.1, 0.2, 0.3], parameters, n_qubits=3, edges=((0, 1),))
        self.assertEqual(states.shape, (4, 8))
        np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-12)

    def test_study_reuses_same_number_of_samples_for_each_topology(self):
        rows = topology_study(n_qubits=3, n_layers=2, topologies=("none", "line", "complete"), samples=2, seed=7)
        self.assertEqual(len(rows), 6)
        counts = {name: sum(row["topology"] == name for row in rows) for name in ("none", "line", "complete")}
        self.assertEqual(counts, {"none": 2, "line": 2, "complete": 2})

    def test_summary_contains_one_row_per_topology(self):
        rows = topology_study(n_qubits=3, n_layers=2, topologies=("none", "line"), samples=2, seed=9)
        summary = summarize(rows)
        self.assertEqual({row["topology"] for row in summary}, {"none", "line"})


if __name__ == "__main__":
    unittest.main()
