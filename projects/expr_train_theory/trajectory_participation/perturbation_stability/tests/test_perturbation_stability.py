import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.trajectory_participation.snapshots import trajectory_snapshots
from projects.expr_train_theory.trajectory_participation.perturbation_stability.study import (
    parameter_perturbation_study,
    perturb_snapshots,
    save_results,
    snapshot_perturbation_study,
    state_overlap_distance,
    summarise_rows,
)


class PerturbationStabilityTests(unittest.TestCase):
    def setUp(self):
        self.states = np.array(
            [[1.0, 0.0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]], dtype=complex
        )

    def test_zero_snapshot_perturbation_is_identity(self):
        rng = np.random.default_rng(1)
        changed = perturb_snapshots(self.states, 0.0, rng)
        np.testing.assert_allclose(changed, self.states)

    def test_perturbed_snapshots_stay_normalized(self):
        rng = np.random.default_rng(2)
        changed = perturb_snapshots(self.states, 0.1, rng)
        np.testing.assert_allclose(np.linalg.norm(changed, axis=1), 1.0, atol=1e-12)

    def test_overlap_distance_is_phase_invariant(self):
        phases = np.exp(1j * np.array([0.3, -0.8]))
        changed = self.states * phases[:, None]
        self.assertAlmostEqual(state_overlap_distance(self.states, changed), 0.0, places=12)

    def test_zero_scale_has_zero_metric_change(self):
        rows = snapshot_perturbation_study(self.states, scales=(0.0,), repeats=4, seed=5)
        self.assertTrue(all(abs(row["delta_d_tp"]) < 1e-12 for row in rows))
        self.assertTrue(all(row["mean_infidelity"] < 1e-12 for row in rows))

    def test_parameter_zero_scale_reproduces_baseline(self):
        parameters = np.zeros((2, 1, 3), dtype=float)
        rows = parameter_perturbation_study(
            [0.2], parameters, n_qubits=1, scales=(0.0,), repeats=3, seed=7
        )
        self.assertTrue(all(abs(row["delta_d_tp"]) < 1e-12 for row in rows))

    def test_summary_groups_by_scale(self):
        rows = snapshot_perturbation_study(
            self.states, scales=(0.0, 0.01), repeats=3, seed=8
        )
        summary = summarise_rows(rows)
        self.assertEqual([row["scale"] for row in summary], [0.0, 0.01])
        self.assertEqual([row["samples"] for row in summary], [3, 3])

    def test_results_are_reproducible_for_fixed_seed(self):
        first = snapshot_perturbation_study(self.states, scales=(0.01,), repeats=3, seed=9)
        second = snapshot_perturbation_study(self.states, scales=(0.01,), repeats=3, seed=9)
        self.assertEqual(first, second)

    def test_save_results_writes_expected_files(self):
        rows = snapshot_perturbation_study(self.states, scales=(0.0,), repeats=1, seed=1)
        summary = summarise_rows(rows)
        with tempfile.TemporaryDirectory() as tmp:
            save_results(tmp, "check", rows, summary, {"test": True})
            self.assertTrue((Path(tmp) / "check_raw.csv").exists())
            self.assertTrue((Path(tmp) / "check_summary.csv").exists())
            self.assertTrue((Path(tmp) / "check_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
