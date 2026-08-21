import tempfile
import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation.metric_profile.profile import (
    default_profiles,
    evaluate_profile,
    mixed_trajectory,
    phase_only_trajectory,
    redundant_ry_trajectory,
    save_profiles,
)


class MetricProfileTests(unittest.TestCase):
    def setUp(self):
        self.X = np.linspace(-0.8, 0.8, 7)

    def test_redundant_ry_has_one_parameter_direction(self):
        row = evaluate_profile(
            "redundant", redundant_ry_trajectory, np.array([0.35, -0.55]), self.X
        )
        self.assertEqual(row.qfim_rank, 1)
        self.assertEqual(row.cfim_rank, 1)
        self.assertEqual(row.qntk_rank, 1)
        self.assertEqual(row.trajectory_rank, 2)

    def test_phase_is_seen_by_qfim_but_not_z_output(self):
        row = evaluate_profile("phase", phase_only_trajectory, np.array([0.9]), self.X)
        self.assertEqual(row.qfim_rank, 1)
        self.assertEqual(row.cfim_rank, 0)
        self.assertEqual(row.qntk_rank, 0)
        self.assertGreater(row.d_tp, 1.0)

    def test_mixed_case_has_two_qfim_directions(self):
        row = evaluate_profile("mixed", mixed_trajectory, np.array([0.6, 1.1]), self.X)
        self.assertEqual(row.qfim_rank, 2)
        self.assertEqual(row.cfim_rank, 1)
        self.assertEqual(row.qntk_rank, 1)
        self.assertEqual(row.trajectory_rank, 2)

    def test_default_profiles_have_unique_names(self):
        rows = default_profiles()
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row.name for row in rows}), 3)

    def test_save_profiles_writes_files(self):
        rows = default_profiles()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, json_path = save_profiles(tmp, rows)
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())


if __name__ == "__main__":
    unittest.main()
