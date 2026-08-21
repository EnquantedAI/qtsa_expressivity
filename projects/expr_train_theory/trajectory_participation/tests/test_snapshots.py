import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation.core import trajectory_participation_dimension
from projects.expr_train_theory.trajectory_participation.snapshots import angle_encoding, trajectory_snapshots
from projects.expr_train_theory.trajectory_participation.study import analyse_toy_qnn_trajectory


class TrajectorySnapshotTests(unittest.TestCase):
    def test_angle_encoding_is_unitary(self):
        unitary = angle_encoding([0.2, -0.4], 2, axis="Y")
        np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(4), atol=1e-12)

    def test_snapshot_count_is_layers_plus_one(self):
        params = np.zeros((3, 2, 3))
        states = trajectory_snapshots([0.2, 0.4], params, n_qubits=2)
        self.assertEqual(states.shape, (4, 4))

    def test_all_snapshots_are_normalized(self):
        rng = np.random.default_rng(2)
        params = rng.normal(size=(4, 2, 3))
        states = trajectory_snapshots([0.3, -0.8], params, n_qubits=2, reupload_axis="Y")
        np.testing.assert_allclose(np.linalg.norm(states, axis=1), np.ones(5), atol=1e-12)

    def test_encoding_only_trajectory_has_dimension_one(self):
        states = trajectory_snapshots([0.1], np.zeros((0, 1, 3)), n_qubits=1)
        self.assertEqual(states.shape, (1, 2))
        result = trajectory_participation_dimension(states)
        self.assertAlmostEqual(result.dimension, 1.0, places=10)

    def test_identity_like_layers_keep_dimension_one_for_zero_features(self):
        params = np.zeros((4, 1, 3))
        states = trajectory_snapshots([0.0], params, n_qubits=1, entangle=False)
        result = trajectory_participation_dimension(states)
        self.assertAlmostEqual(result.dimension, 1.0, places=10)

    def test_reuploading_can_change_the_trajectory(self):
        params = np.zeros((2, 1, 3))
        plain = trajectory_snapshots([0.7], params, n_qubits=1, reupload_axis=None, entangle=False)
        reuploaded = trajectory_snapshots([0.7], params, n_qubits=1, reupload_axis="Y", entangle=False)
        d_plain = trajectory_participation_dimension(plain).dimension
        d_reuploaded = trajectory_participation_dimension(reuploaded).dimension
        self.assertAlmostEqual(d_plain, 1.0, places=10)
        self.assertGreater(d_reuploaded, d_plain + 1e-6)

    def test_study_normalization_is_bounded(self):
        rng = np.random.default_rng(5)
        params = rng.normal(size=(3, 2, 3))
        result = analyse_toy_qnn_trajectory([0.2, 0.9], params, n_qubits=2)
        self.assertGreaterEqual(result.normalized_d_tp, 0.0)
        self.assertLessEqual(result.normalized_d_tp, 1.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
