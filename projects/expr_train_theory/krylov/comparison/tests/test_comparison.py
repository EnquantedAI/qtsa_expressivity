import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.krylov.comparison.study import (
    compare_hamiltonian_and_unitary_krylov,
    principal_angles,
    projector_distance,
    run_time_step_study,
)
from projects.expr_train_theory.krylov.models import path_hamiltonian


class ComparisonTests(unittest.TestCase):
    def test_identical_subspaces_have_zero_distance(self) -> None:
        basis = np.eye(3, dtype=np.complex128)
        self.assertTrue(np.allclose(principal_angles(basis, basis), 0.0))
        self.assertAlmostEqual(projector_distance(basis, basis), 0.0)

    def test_basis_change_does_not_change_subspace(self) -> None:
        first = np.eye(3, dtype=np.complex128)[:, :2]
        rotation = np.array(
            [[1.0, 1.0j], [1.0j, 1.0]], dtype=np.complex128
        ) / np.sqrt(2.0)
        second = first @ rotation
        self.assertAlmostEqual(projector_distance(first, second), 0.0, places=10)

    def test_generic_time_step_recovers_full_invariant_subspace(self) -> None:
        hamiltonian = path_hamiltonian(4)
        initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        result = compare_hamiltonian_and_unitary_krylov(
            hamiltonian, initial, time_step=0.3
        )
        self.assertEqual(result.hamiltonian_dimension, 4)
        self.assertEqual(result.unitary_dimension, 4)
        self.assertLess(result.projector_distance, 1e-8)

    def test_phase_aliasing_can_reduce_unitary_dimension(self) -> None:
        hamiltonian = np.diag([0.0, 1.0, 2.0])
        initial = np.ones(3, dtype=np.complex128) / np.sqrt(3.0)
        result = compare_hamiltonian_and_unitary_krylov(
            hamiltonian, initial, time_step=2.0 * np.pi
        )
        self.assertEqual(result.hamiltonian_dimension, 3)
        self.assertEqual(result.unitary_dimension, 1)
        self.assertGreater(result.projector_distance, 1.0)

    def test_study_writes_output_files(self) -> None:
        hamiltonian = path_hamiltonian(3)
        initial = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        with tempfile.TemporaryDirectory() as directory:
            results = run_time_step_study(
                hamiltonian,
                initial,
                [0.2, 0.5],
                output_directory=directory,
            )
            self.assertEqual(len(results), 2)
            self.assertTrue(
                (Path(directory) / "hamiltonian_unitary_comparison.csv").exists()
            )
            self.assertTrue(
                (
                    Path(directory)
                    / "hamiltonian_unitary_comparison_metadata.json"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
