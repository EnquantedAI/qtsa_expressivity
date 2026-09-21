import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pennylane as qml
except ImportError:
    qml = None

from projects.expr_train_theory.trajectory_participation.ancilla_coupling.metrics import (
    ancilla_growth,
    ancilla_reduced_state,
    ancilla_state_diagnostics,
    safe_pearson,
)
from projects.expr_train_theory.trajectory_participation.ancilla_coupling.study import (
    evaluate_coupling,
    run_ancilla_coupling_study,
    save_results,
)
from projects.expr_train_theory.trajectory_participation.native_shared_qnn_study.study import (
    NativeArchitectureConfig,
)


def _toy_snapshot_fn(inputs, weights, *, n_qubits, fm_style, reup_style):
    n_layers = np.asarray(weights).shape[0]
    states = []
    for depth in range(n_layers + 1):
        state = np.zeros(2**n_qubits, dtype=np.complex128)
        if n_qubits >= 2 and depth > 0:
            # Entangle the first system wire with the final ancilla wire while
            # varying the Schmidt weight with depth.
            angle = min(np.pi / 4, 0.2 * depth)
            state[0] = np.cos(angle)
            state[-1] = np.sin(angle)
        else:
            angle = 0.08 * depth
            state[0] = np.cos(angle)
            state[-1] = np.sin(angle)
        states.append(state)
    return np.asarray(states)


class AncillaMetricTests(unittest.TestCase):
    def test_product_state_has_zero_entropy_and_unit_purity(self):
        state = np.zeros(4, dtype=np.complex128)
        state[0] = 1.0
        diagnostics = ancilla_state_diagnostics(state, n_qubits=2, n_ancilla=1)
        self.assertAlmostEqual(diagnostics["ancilla_entropy"], 0.0, places=12)
        self.assertAlmostEqual(diagnostics["ancilla_purity"], 1.0, places=12)
        self.assertAlmostEqual(diagnostics["ancilla_participation_dimension"], 1.0, places=12)
        self.assertAlmostEqual(diagnostics["ancilla_mixedness_fraction"], 0.0, places=12)

    def test_bell_state_has_maximally_mixed_single_ancilla(self):
        state = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        rho = ancilla_reduced_state(state, n_qubits=2, n_ancilla=1)
        np.testing.assert_allclose(rho, np.eye(2) / 2.0, atol=1e-12)
        diagnostics = ancilla_state_diagnostics(state, n_qubits=2, n_ancilla=1)
        self.assertAlmostEqual(diagnostics["ancilla_entropy"], np.log(2.0), places=12)
        self.assertAlmostEqual(diagnostics["ancilla_entropy_fraction"], 1.0, places=12)
        self.assertAlmostEqual(diagnostics["ancilla_purity"], 0.5, places=12)
        self.assertAlmostEqual(diagnostics["ancilla_participation_dimension"], 2.0, places=12)
        self.assertAlmostEqual(diagnostics["ancilla_mixedness_fraction"], 1.0, places=12)

    def test_zero_ancilla_is_exact_reference(self):
        state = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
        diagnostics = ancilla_state_diagnostics(state, n_qubits=1, n_ancilla=0)
        self.assertEqual(diagnostics["ancilla_entropy"], 0.0)
        self.assertEqual(diagnostics["ancilla_purity"], 1.0)
        self.assertEqual(diagnostics["ancilla_participation_dimension"], 1.0)
        self.assertEqual(diagnostics["ancilla_mixedness_fraction"], 0.0)

    def test_reduced_state_is_global_phase_invariant(self):
        state = np.array([1.0, 0.0, 0.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
        rho = ancilla_reduced_state(state, 2, 1)
        shifted = ancilla_reduced_state(np.exp(0.73j) * state, 2, 1)
        np.testing.assert_allclose(rho, shifted, atol=1e-12)

    def test_safe_pearson_returns_none_for_constant_series(self):
        self.assertIsNone(safe_pearson([1, 1, 1], [0, 1, 2]))
        self.assertAlmostEqual(safe_pearson([0, 1, 2], [0, 2, 4]), 1.0, places=12)


class AncillaCouplingStudyTests(unittest.TestCase):
    def test_evaluate_coupling_combines_depth_diagnostics(self):
        config = NativeArchitectureConfig(3, 1, 1, "Y", None)
        weights = np.zeros((3, 2, 3))
        states, final, growth, coupling = evaluate_coupling(
            config,
            np.array([0.2]),
            weights,
            snapshot_fn=_toy_snapshot_fn,
        )
        self.assertEqual(states.shape, (4, 4))
        self.assertEqual(len(growth), 4)
        self.assertIn("d_tp_equal", final)
        self.assertIn("ancilla_entropy", final)
        self.assertGreater(final["ancilla_entropy"], 0.0)
        self.assertIsNotNone(coupling["corr_dtp_equal_entropy"])

    def test_study_keeps_zero_ancilla_correlations_undefined(self):
        rows, summary, growth, growth_summary = run_ancilla_coupling_study(
            layers=(2,),
            n_features=1,
            ancilla_counts=(0, 1),
            feature_maps=("Y",),
            reupload_styles=(None,),
            samples=1,
            seed=9,
            snapshot_fn=_toy_snapshot_fn,
        )
        self.assertEqual(len(rows), 2)
        zero = next(row for row in rows if row["n_ancilla"] == 0)
        one = next(row for row in rows if row["n_ancilla"] == 1)
        self.assertIsNone(zero["corr_dtp_equal_entropy"])
        self.assertEqual(zero["final_ancilla_purity"], 1.0)
        self.assertGreater(one["final_ancilla_entropy"], 0.0)
        self.assertEqual(len(summary), 2)
        self.assertGreater(len(growth), len(rows))
        self.assertGreater(len(growth_summary), len(summary))

    def test_save_results_contract(self):
        rows, summary, growth, growth_summary = run_ancilla_coupling_study(
            layers=(2,),
            n_features=1,
            ancilla_counts=(1,),
            feature_maps=("Y",),
            reupload_styles=(None,),
            samples=1,
            snapshot_fn=_toy_snapshot_fn,
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_results(
                tmp,
                rows,
                summary,
                growth,
                growth_summary,
                metadata={"kind": "test"},
            )
            names = {path.name for path in Path(tmp).iterdir()}
        self.assertEqual(
            names,
            {
                "ancilla_coupling_raw.csv",
                "ancilla_coupling_summary.csv",
                "ancilla_coupling_growth_raw.csv",
                "ancilla_coupling_growth_summary.csv",
                "ancilla_coupling_metadata.json",
            },
        )


@unittest.skipIf(qml is None, "PennyLane is not available")
class PennyLaneAncillaCouplingIntegrationTests(unittest.TestCase):
    def test_small_native_coupling_study_runs(self):
        rows, summary, growth, growth_summary = run_ancilla_coupling_study(
            layers=(1,),
            n_features=1,
            ancilla_counts=(1,),
            feature_maps=("Y",),
            reupload_styles=(None,),
            samples=1,
            seed=17,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(summary), 1)
        self.assertEqual(len(growth), 2)
        self.assertEqual(len(growth_summary), 2)
        self.assertTrue(np.isfinite(rows[0]["final_ancilla_purity"]))


if __name__ == "__main__":
    unittest.main()
