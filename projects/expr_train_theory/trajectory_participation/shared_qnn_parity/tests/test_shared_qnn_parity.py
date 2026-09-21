import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pennylane as qml
except ImportError:
    qml = None

from projects.expr_train_theory.trajectory_participation.shared_qnn_parity.study import (
    DEFAULT_PARITY_CONFIGS,
    ParityConfig,
    compare_trajectories,
    evaluate_config,
    phase_aligned_state_error,
    run_parity_study,
    save_results,
    state_fidelity,
    summarize_rows,
)


def _toy_snapshot_fn(inputs, weights, *, n_qubits, fm_style, reup_style):
    if n_qubits != 1:
        raise ValueError("toy helper uses one qubit")
    angle = float(np.asarray(inputs).reshape(-1)[0] + np.sum(weights))
    state0 = np.array([1.0, 0.0], dtype=np.complex128)
    state1 = np.array(
        [np.cos(angle / 2.0), np.sin(angle / 2.0)], dtype=np.complex128
    )
    return np.asarray([state0, state1])


def _phase_shifted_toy_snapshot_fn(inputs, weights, *, n_qubits, fm_style, reup_style):
    states = _toy_snapshot_fn(
        inputs,
        weights,
        n_qubits=n_qubits,
        fm_style=fm_style,
        reup_style=reup_style,
    )
    phases = np.exp(1j * np.array([0.37, -0.91]))[:, None]
    return states * phases


class StateParityHelpersTests(unittest.TestCase):
    def test_fidelity_ignores_global_phase(self):
        state = np.array([1.0, 1.0j]) / np.sqrt(2.0)
        shifted = np.exp(0.43j) * state
        self.assertAlmostEqual(state_fidelity(state, shifted), 1.0, places=12)

    def test_phase_aligned_error_ignores_global_phase(self):
        state = np.array([1.0, -1.0j]) / np.sqrt(2.0)
        shifted = np.exp(-1.17j) * state
        self.assertLess(phase_aligned_state_error(shifted, state), 1e-12)

    def test_trajectory_comparison_is_snapshot_phase_invariant(self):
        reference = np.eye(2, dtype=np.complex128)
        native = reference * np.exp(1j * np.array([0.2, -0.8]))[:, None]
        result = compare_trajectories(native, reference, tolerance=1e-10)
        self.assertTrue(result["passed"])
        self.assertLess(result["d_tp_abs_error"], 1e-12)
        self.assertLess(result["max_gram_magnitude_error"], 1e-12)

    def test_trajectory_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            compare_trajectories(np.eye(2), np.ones((3, 2)))


class ParityStudyTests(unittest.TestCase):
    def test_default_matrix_covers_shared_qnn_choices(self):
        self.assertEqual(
            {config.fm_style for config in DEFAULT_PARITY_CONFIGS},
            {"X", "Y", "Z", "zzfm", "iqp"},
        )
        self.assertEqual(
            {config.reup_style for config in DEFAULT_PARITY_CONFIGS},
            {None, "X", "Y", "Z"},
        )
        self.assertGreater(len({config.n_qubits for config in DEFAULT_PARITY_CONFIGS}), 1)
        self.assertGreater(len({config.n_layers for config in DEFAULT_PARITY_CONFIGS}), 1)

    def test_evaluate_config_uses_matched_inputs_and_weights(self):
        config = ParityConfig(1, 1, "Y", None, 1)
        rows = evaluate_config(
            config,
            samples=3,
            seed=17,
            native_fn=_phase_shifted_toy_snapshot_fn,
            reference_fn=_toy_snapshot_fn,
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertTrue(all(row["min_snapshot_fidelity"] > 1 - 1e-12 for row in rows))

    def test_summary_and_save_results(self):
        config = ParityConfig(1, 1, "Y", None, 1)
        rows = evaluate_config(
            config,
            samples=2,
            seed=5,
            native_fn=_phase_shifted_toy_snapshot_fn,
            reference_fn=_toy_snapshot_fn,
        )
        summary = summarize_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertTrue(summary[0]["all_passed"])

        with tempfile.TemporaryDirectory() as tmp:
            save_results(
                tmp,
                rows,
                summary,
                configs=(config,),
                samples=2,
                seed=5,
                tolerance=1e-8,
            )
            names = {path.name for path in Path(tmp).iterdir()}
        self.assertEqual(
            names,
            {
                "shared_qnn_parity_raw.csv",
                "shared_qnn_parity_summary.csv",
                "shared_qnn_parity_metadata.json",
            },
        )


@unittest.skipIf(qml is None, "PennyLane is not available")
class PennyLaneParityIntegrationTests(unittest.TestCase):
    def test_default_parity_matrix_matches_shared_model(self):
        rows, summary = run_parity_study(
            configs=DEFAULT_PARITY_CONFIGS,
            samples=1,
            seed=71,
            tolerance=1e-8,
        )
        self.assertEqual(len(rows), len(DEFAULT_PARITY_CONFIGS))
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertTrue(all(row["all_passed"] for row in summary))


if __name__ == "__main__":
    unittest.main()
