from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import pennylane as qml
except ImportError:
    qml = None

from projects.expr_train_theory.trajectory_participation.real_data_stability.study import (
    _crossed_bootstrap_means,
    matched_effect_rows,
    run_real_data_stability,
    save_results,
    summarize_effect_stability,
)


METRIC = "final_d_tp_equal_normalized"


def _raw_row(*, window, seed, layers, ancilla, value):
    return {
        "dataset": "toy",
        "split": "test",
        "window_index": window,
        "weight_seed": seed,
        "seed": seed,
        "weight_sample": 0,
        "n_layers": layers,
        "n_features": 2,
        "n_ancilla": ancilla,
        "n_qubits": 2 + ancilla,
        "fm_style": "Y",
        "reup_style": "none",
        METRIC: value,
    }


def _effect_row(window, seed, value):
    return {
        "effect_type": "ancilla",
        "dataset": "toy",
        "split": "test",
        "window_index": window,
        "weight_seed": seed,
        "weight_sample": 0,
        "n_features": 2,
        "fm_style": "Y",
        "reup_style": "none",
        "n_layers": 2,
        "baseline_n_ancilla": 0,
        "comparison_n_ancilla": 1,
        "n_ancilla": None,
        "baseline_n_layers": None,
        "comparison_n_layers": None,
        f"delta_{METRIC}": value,
    }


def _toy_snapshot_fn(inputs, weights, *, n_qubits, fm_style, reup_style):
    inputs = np.asarray(inputs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    states = []
    for depth in range(weights.shape[0] + 1):
        angle = 0.05 * float(np.sum(inputs)) + 0.02 * float(np.sum(weights[:depth]))
        angle += 0.01 * n_qubits * depth
        state = np.zeros(2**n_qubits, dtype=np.complex128)
        state[0] = np.cos(angle)
        state[-1] = np.sin(angle)
        states.append(state)
    return np.asarray(states)


class MatchedEffectTests(unittest.TestCase):
    def test_ancilla_and_depth_deltas_are_matched_before_averaging(self):
        rows = []
        for window in (0, 1):
            for seed in (11, 12):
                base = 10 * window + seed
                rows.extend(
                    [
                        _raw_row(window=window, seed=seed, layers=1, ancilla=0, value=base),
                        _raw_row(window=window, seed=seed, layers=1, ancilla=1, value=base + 2),
                        _raw_row(window=window, seed=seed, layers=2, ancilla=0, value=base + 3),
                        _raw_row(window=window, seed=seed, layers=2, ancilla=1, value=base + 7),
                    ]
                )
        effects = matched_effect_rows(rows, metrics=(METRIC,), baseline_ancilla=0, baseline_depth=1)
        ancilla_l1 = [r for r in effects if r["effect_type"] == "ancilla" and r["n_layers"] == 1]
        depth_a0 = [r for r in effects if r["effect_type"] == "depth" and r["n_ancilla"] == 0]
        self.assertEqual({r[f"delta_{METRIC}"] for r in ancilla_l1}, {2.0})
        self.assertEqual({r[f"delta_{METRIC}"] for r in depth_a0}, {3.0})

    def test_missing_baseline_is_rejected(self):
        rows = [_raw_row(window=0, seed=1, layers=2, ancilla=1, value=1.0)]
        with self.assertRaises(ValueError):
            matched_effect_rows(rows, metrics=(METRIC,), baseline_ancilla=0)
        with self.assertRaises(ValueError):
            matched_effect_rows(rows, metrics=(METRIC,), baseline_ancilla=1, baseline_depth=1)


class StabilitySummaryTests(unittest.TestCase):
    def test_strictly_positive_effect_is_classified_positive(self):
        rows = [_effect_row(w, s, 0.2 + 0.01 * w) for w in range(3) for s in (1, 2, 3)]
        summary = summarize_effect_stability(
            rows,
            metrics=(METRIC,),
            bootstrap_samples=200,
            bootstrap_seed=7,
        )
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["classification"], "stable_positive")
        self.assertGreater(summary[0]["ci_low"], 0.0)

    def test_mixed_sign_effect_is_unresolved(self):
        values = {(0, 1): -1.0, (0, 2): 1.0, (1, 1): 1.0, (1, 2): -1.0}
        rows = [_effect_row(w, s, value) for (w, s), value in values.items()]
        summary = summarize_effect_stability(
            rows,
            metrics=(METRIC,),
            bootstrap_samples=300,
            bootstrap_seed=9,
        )
        self.assertEqual(summary[0]["classification"], "unresolved")
        self.assertLessEqual(summary[0]["ci_low"], 0.0)
        self.assertGreaterEqual(summary[0]["ci_high"], 0.0)

    def test_crossed_bootstrap_is_seeded(self):
        rows = [_effect_row(w, s, 0.1 * (w + s)) for w in range(3) for s in (5, 6)]
        first = _crossed_bootstrap_means(rows, METRIC, samples=50, rng=np.random.default_rng(123))
        second = _crossed_bootstrap_means(rows, METRIC, samples=50, rng=np.random.default_rng(123))
        np.testing.assert_allclose(first, second)

    def test_summary_separates_window_and_weight_draw_variability(self):
        rows = []
        for w in (0, 1, 2):
            for s in (1, 2, 3):
                rows.append(_effect_row(w, s, float(w)))
        summary = summarize_effect_stability(rows, metrics=(METRIC,), bootstrap_samples=50)
        self.assertGreater(summary[0]["window_mean_std"], 0.0)
        self.assertAlmostEqual(summary[0]["weight_draw_mean_std"], 0.0, places=12)


class RealDataStabilityEndToEndTests(unittest.TestCase):
    def test_repeat_seeds_reuse_exact_same_real_windows(self):
        raw, effects, summary, metadata = run_real_data_stability(
            dataset_names=("NARMA10_Chaotic",),
            n_windows=3,
            layers=(1, 2),
            ancilla_counts=(0, 1),
            feature_maps=("Y",),
            reupload_styles=(None,),
            repeat_seeds=(11, 12),
            bootstrap_samples=30,
            snapshot_fn=_toy_snapshot_fn,
        )
        self.assertEqual(metadata["selected_window_indices"]["NARMA10_Chaotic"], [0, 31, 62])
        self.assertEqual({row["window_index"] for row in raw if row["weight_seed"] == 11}, {0, 31, 62})
        self.assertEqual({row["window_index"] for row in raw if row["weight_seed"] == 12}, {0, 31, 62})
        self.assertTrue(effects)
        self.assertTrue(summary)
        self.assertEqual(metadata["repeat_seeds"], [11, 12])

    def test_save_results_contract(self):
        raw, effects, summary, metadata = run_real_data_stability(
            dataset_names=("Mackey_Glass_tau_30",),
            n_windows=2,
            layers=(1, 2),
            ancilla_counts=(0, 1),
            feature_maps=("Y",),
            reupload_styles=(None,),
            repeat_seeds=(3, 4),
            bootstrap_samples=20,
            snapshot_fn=_toy_snapshot_fn,
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_results(tmp, raw, effects, summary, metadata)
            names = {path.name for path in Path(tmp).iterdir()}
        self.assertEqual(
            names,
            {
                "real_data_stability_raw.csv",
                "real_data_stability_effects.csv",
                "real_data_stability_summary.csv",
                "real_data_stability_metadata.json",
            },
        )


@unittest.skipIf(qml is None, "PennyLane is not available")
class PennyLaneRealDataStabilityIntegrationTests(unittest.TestCase):
    def test_small_native_stability_run(self):
        raw, effects, summary, _metadata = run_real_data_stability(
            dataset_names=("NARMA10_Chaotic",),
            n_windows=1,
            layers=(1, 2),
            ancilla_counts=(0, 1),
            feature_maps=("Y",),
            reupload_styles=(None,),
            repeat_seeds=(5, 6),
            bootstrap_samples=10,
        )
        self.assertTrue(raw)
        self.assertTrue(effects)
        self.assertTrue(summary)
        self.assertTrue(all(np.isfinite(row["mean_delta"]) for row in summary))


if __name__ == "__main__":
    unittest.main()
