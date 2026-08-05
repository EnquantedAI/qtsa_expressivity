from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.krylov.circuit_ensemble.study import EnsembleConfig, EnsembleRecord
from projects.expr_train_theory.krylov.metric_screening.screening import (
    ScreeningConfig,
    eta_squared,
    run_screening,
    screen_records,
)


def _record(depth: int, topology: str, orbit: float, entropy: float) -> EnsembleRecord:
    return EnsembleRecord(
        n_qubits=2,
        depth=depth,
        topology=topology,
        sample=0,
        seed=1,
        hilbert_dimension=4,
        layer_orbit_dimension=int(round(4 * orbit)),
        repeated_cycle_dimension=4,
        layer_orbit_fraction=orbit,
        repeated_cycle_fraction=1.0,
        projector_distance=2.0 * (1.0 - orbit),
        max_principal_angle=0.0,
        smallest_subspace_overlap=1.0,
        layer_gram_condition=1.0 + depth,
        final_mean_single_qubit_entropy=entropy,
        max_mean_single_qubit_entropy=entropy,
    )


class EtaSquaredTests(unittest.TestCase):
    def test_perfect_group_separation(self) -> None:
        values = np.array([0.0, 0.0, 1.0, 1.0])
        groups = np.array(["a", "a", "b", "b"])
        self.assertAlmostEqual(eta_squared(values, groups), 1.0)

    def test_constant_values_return_zero(self) -> None:
        self.assertEqual(eta_squared(np.ones(4), np.array([0, 0, 1, 1])), 0.0)


class ScreeningTests(unittest.TestCase):
    def test_constant_and_redundant_metrics_are_flagged(self) -> None:
        records = [
            _record(1, "none", 0.25, 0.0),
            _record(2, "none", 0.50, 0.0),
            _record(3, "full", 0.75, 1.0),
            _record(4, "full", 1.00, 1.0),
        ]
        rows = screen_records(records)
        by_name = {row["metric"]: row for row in rows}
        self.assertTrue(by_name["repeated_cycle_fraction"]["constant_in_sample"])
        self.assertTrue(by_name["layer_orbit_fraction"]["highly_redundant_in_sample"])
        self.assertAlmostEqual(by_name["final_mean_single_qubit_entropy"]["topology_eta_squared"], 1.0)

    def test_unknown_metric_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            screen_records([_record(1, "none", 0.5, 0.0)], metrics=("missing",))

    def test_run_writes_results(self) -> None:
        config = ScreeningConfig(
            ensemble=EnsembleConfig(
                n_qubits=2,
                depths=(1,),
                topologies=("none", "linear"),
                samples=2,
                seed=7,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            _, rows = run_screening(config, directory)
            self.assertTrue(rows)
            self.assertTrue((Path(directory) / "metric_screening.csv").exists())
            self.assertTrue((Path(directory) / "metric_screening_metadata.json").exists())
            self.assertTrue((Path(directory) / "ensemble" / "circuit_ensemble_raw.csv").exists())


if __name__ == "__main__":
    unittest.main()
