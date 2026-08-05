import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.krylov.experiments.benchmark import (
    BenchmarkCase,
    run_benchmark,
    save_benchmark,
)
from projects.expr_train_theory.krylov.models import basis_state


class KrylovBenchmarkTests(unittest.TestCase):
    def test_eigenstate_case_stays_one_dimensional(self) -> None:
        case = BenchmarkCase(
            name="eigenstate",
            hamiltonian=np.diag([0.0, 1.0]).astype(np.complex128),
            initial_state=basis_state(1, 2),
            description="test",
        )
        rows, summaries, _ = run_benchmark([case], times=[0.0, 0.5, 1.0])

        self.assertEqual(len(rows), 3)
        self.assertEqual(summaries[0]["krylov_dimension"], 1)
        self.assertAlmostEqual(summaries[0]["max_spread_complexity"], 0.0, places=12)
        self.assertAlmostEqual(summaries[0]["max_state_error"], 0.0, places=12)

    def test_full_krylov_space_reproduces_exact_evolution(self) -> None:
        case = BenchmarkCase(
            name="pauli_x",
            hamiltonian=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
            initial_state=basis_state(0, 2),
            description="test",
        )
        _, summaries, _ = run_benchmark([case], times=np.linspace(0.0, 2.0, 9))
        self.assertLess(summaries[0]["max_state_error"], 1e-10)
        self.assertEqual(summaries[0]["krylov_dimension"], 2)

    def test_save_benchmark_writes_all_files(self) -> None:
        case = BenchmarkCase(
            name="simple",
            hamiltonian=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
            initial_state=basis_state(0, 2),
            description="test",
        )
        rows, summaries, settings = run_benchmark([case], times=[0.0, 1.0])
        with tempfile.TemporaryDirectory() as directory:
            paths = save_benchmark(directory, rows, summaries, settings)
            self.assertTrue(all(Path(path).is_file() for path in paths))

    def test_empty_time_grid_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_benchmark(times=[])


if __name__ == "__main__":
    unittest.main()
