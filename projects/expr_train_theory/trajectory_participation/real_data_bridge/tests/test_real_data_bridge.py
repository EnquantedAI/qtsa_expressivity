from pathlib import Path
import tempfile
import unittest

import numpy as np

from projects.expr_train_theory.trajectory_participation.real_data_bridge.data import (
    PROJECT_DATASET_NAMES,
    load_project_dataset,
    project_dataset_path,
    select_window_indices,
    select_windows,
)
from projects.expr_train_theory.trajectory_participation.real_data_bridge.study import (
    run_real_data_study,
    save_results,
)


class RealDataLoadingTests(unittest.TestCase):
    def test_project_dataset_files_exist(self):
        for name in PROJECT_DATASET_NAMES:
            self.assertTrue(project_dataset_path(name, "test").is_file())

    def test_narma_test_dataset_shape(self):
        dataset = load_project_dataset("NARMA10_Chaotic", "test")
        self.assertEqual(dataset.inputs.shape, (63, 4))
        self.assertEqual(dataset.targets.shape, (63,))
        self.assertTrue(np.all(np.isfinite(dataset.inputs)))

    def test_mackey_glass_test_dataset_shape(self):
        dataset = load_project_dataset("Mackey_Glass_tau_30", "test")
        self.assertEqual(dataset.inputs.shape, (63, 4))
        self.assertEqual(dataset.targets.shape, (63,))
        self.assertTrue(np.all(np.isfinite(dataset.targets)))

    def test_even_selection_is_deterministic_and_spans_split(self):
        self.assertEqual(select_window_indices(63, 4, mode="even").tolist(), [0, 21, 41, 62])

    def test_random_selection_is_seeded(self):
        first = select_window_indices(63, 8, mode="random", seed=17)
        second = select_window_indices(63, 8, mode="random", seed=17)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 8)

    def test_selected_windows_keep_targets_and_indices(self):
        dataset = load_project_dataset("NARMA10_Chaotic", "test")
        selected = select_windows(dataset, 3, mode="even")
        self.assertEqual([row["window_index"] for row in selected], [0, 31, 62])
        np.testing.assert_allclose(selected[1]["inputs"], dataset.inputs[31])
        self.assertEqual(selected[1]["target"], float(dataset.targets[31]))

    def test_unknown_dataset_or_split_is_rejected(self):
        with self.assertRaises(ValueError):
            project_dataset_path("unknown", "test")
        with self.assertRaises(ValueError):
            project_dataset_path("NARMA10_Chaotic", "holdout")


class RealDataStudyTests(unittest.TestCase):
    @staticmethod
    def fake_snapshot_fn(inputs, weights, *, n_qubits, **_kwargs):
        inputs = np.asarray(inputs, dtype=float)
        weights = np.asarray(weights, dtype=float)
        states = []
        ancilla_dim = 2 ** max(0, n_qubits - inputs.size)
        for depth in range(weights.shape[0] + 1):
            angle = float(np.sum(inputs) + np.sum(weights[:depth])) * 0.1
            system = np.zeros(2 ** inputs.size, dtype=np.complex128)
            system[0] = np.cos(angle)
            system[1] = np.sin(angle)
            ancilla = np.zeros(ancilla_dim, dtype=np.complex128)
            ancilla[0] = 1.0
            states.append(np.kron(system, ancilla))
        return np.asarray(states)

    def test_end_to_end_bridge_uses_real_windows(self):
        rows, summary, growth, growth_summary, metadata = run_real_data_study(
            dataset_names=("NARMA10_Chaotic",),
            split="test",
            n_windows=3,
            layers=(1,),
            ancilla_counts=(0, 1),
            feature_maps=("Y",),
            reupload_styles=(None,),
            weight_samples=1,
            seed=11,
            snapshot_fn=self.fake_snapshot_fn,
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(summary), 2)
        self.assertEqual(len(growth), 12)
        self.assertEqual(len(growth_summary), 4)
        self.assertEqual(metadata["n_features"], 4)
        self.assertEqual(metadata["selected_window_indices"]["NARMA10_Chaotic"], [0, 31, 62])
        self.assertFalse(metadata["weights_trained"])
        self.assertFalse(metadata["targets_used_in_metrics"])
        self.assertTrue(all(row["dataset"] == "NARMA10_Chaotic" for row in rows))

    def test_save_results_contract(self):
        rows, summary, growth, growth_summary, metadata = run_real_data_study(
            dataset_names=("Mackey_Glass_tau_30",),
            n_windows=2,
            layers=(1,),
            ancilla_counts=(0,),
            feature_maps=("Y",),
            reupload_styles=(None,),
            snapshot_fn=self.fake_snapshot_fn,
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_results(tmp, rows, summary, growth, growth_summary, metadata)
            names = {path.name for path in Path(tmp).iterdir()}
            self.assertEqual(
                names,
                {
                    "real_data_raw.csv",
                    "real_data_summary.csv",
                    "real_data_growth_raw.csv",
                    "real_data_growth_summary.csv",
                    "real_data_metadata.json",
                },
            )


if __name__ == "__main__":
    unittest.main()
