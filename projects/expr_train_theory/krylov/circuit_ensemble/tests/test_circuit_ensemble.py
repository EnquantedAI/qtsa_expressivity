import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.krylov.circuit_ensemble.study import (
    EnsembleConfig,
    build_layers,
    mean_single_qubit_entropy,
    run_ensemble_study,
    summarise_records,
)


class CircuitEnsembleTests(unittest.TestCase):
    def test_none_topology_builds_unitary_layers(self):
        parameters = np.zeros((2, 2, 3))
        layers = build_layers(parameters, 2, "none")
        self.assertEqual(len(layers), 2)
        for layer in layers:
            np.testing.assert_allclose(layer.conj().T @ layer, np.eye(4), atol=1e-12)

    def test_all_supported_topologies_are_unitary(self):
        rng = np.random.default_rng(3)
        parameters = rng.normal(size=(2, 3, 3))
        for topology in ("none", "linear", "ring", "full"):
            for layer in build_layers(parameters, 3, topology):
                np.testing.assert_allclose(layer.conj().T @ layer, np.eye(8), atol=1e-12)

    def test_unknown_topology_is_rejected(self):
        with self.assertRaises(ValueError):
            build_layers(np.zeros((1, 2, 3)), 2, "random")

    def test_reproducible_for_fixed_seed(self):
        config = EnsembleConfig(2, (1, 2), ("none", "linear"), samples=3, seed=11)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run_ensemble_study(config, first)
            b = run_ensemble_study(config, second)
        self.assertEqual(a, b)

    def test_summary_has_one_row_per_depth_and_topology(self):
        config = EnsembleConfig(2, (1, 2), ("none", "linear"), samples=2, seed=4)
        with tempfile.TemporaryDirectory() as directory:
            records = run_ensemble_study(config, directory)
            rows = summarise_records(records)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(0.0 <= row["layer_full_fraction"] <= 1.0 for row in rows))
        self.assertTrue(all(0.0 <= row["cycle_full_fraction"] <= 1.0 for row in rows))

    def test_product_state_has_zero_single_qubit_entropy(self):
        state = np.zeros(8, dtype=np.complex128)
        state[0] = 1.0
        self.assertAlmostEqual(mean_single_qubit_entropy(state, 3), 0.0, places=12)

    def test_bell_state_has_unit_single_qubit_entropy(self):
        state = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        self.assertAlmostEqual(mean_single_qubit_entropy(state, 2), 1.0, places=12)

    def test_result_files_are_written(self):
        config = EnsembleConfig(2, (1,), ("none",), samples=1, seed=8)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_ensemble_study(config, output)
            self.assertTrue((output / "circuit_ensemble_raw.csv").exists())
            self.assertTrue((output / "circuit_ensemble_summary.csv").exists())
            self.assertTrue((output / "circuit_ensemble_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
