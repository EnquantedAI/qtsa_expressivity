import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.expr_train_theory.krylov.models import basis_state, path_hamiltonian
from projects.expr_train_theory.krylov.stability.study import (
    perturb_hermitian,
    perturb_state,
    run_stability_study,
    save_stability_study,
)


class StabilityStudyTests(unittest.TestCase):
    def test_zero_scale_keeps_inputs(self) -> None:
        rng = np.random.default_rng(1)
        hamiltonian = path_hamiltonian(3)
        state = basis_state(0, 3)
        np.testing.assert_allclose(perturb_hermitian(hamiltonian, 0.0, rng), hamiltonian)
        np.testing.assert_allclose(perturb_state(state, 0.0, rng), state)

    def test_perturbed_hamiltonian_is_hermitian(self) -> None:
        rng = np.random.default_rng(2)
        perturbed = perturb_hermitian(path_hamiltonian(4), 0.1, rng)
        np.testing.assert_allclose(perturbed, perturbed.conj().T, atol=1e-12)

    def test_perturbed_state_is_normalised(self) -> None:
        rng = np.random.default_rng(3)
        state = perturb_state(basis_state(0, 4), 0.2, rng)
        self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=12)

    def test_study_is_reproducible(self) -> None:
        kwargs = dict(
            hamiltonian=path_hamiltonian(3),
            initial_state=basis_state(0, 3),
            perturbation_scales=(0.0, 1e-2),
            repeats=3,
            times=(0.0, 0.5, 1.0),
            seed=12,
        )
        rows_a, summaries_a, _ = run_stability_study(**kwargs)
        rows_b, summaries_b, _ = run_stability_study(**kwargs)
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(summaries_a, summaries_b)

    def test_zero_scale_has_zero_variance(self) -> None:
        _, summaries, _ = run_stability_study(
            path_hamiltonian(3),
            basis_state(0, 3),
            perturbation_scales=(0.0,),
            repeats=4,
            times=(0.0, 0.5),
            seed=1,
        )
        self.assertAlmostEqual(float(summaries[0]["krylov_dimension_std"]), 0.0)
        self.assertAlmostEqual(float(summaries[0]["max_spread_complexity_std"]), 0.0)

    def test_results_can_be_saved(self) -> None:
        rows, summaries, settings = run_stability_study(
            path_hamiltonian(3),
            basis_state(0, 3),
            perturbation_scales=(0.0,),
            repeats=1,
            times=(0.0, 0.5),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = save_stability_study(directory, rows, summaries, settings)
            self.assertTrue(all(Path(path).exists() for path in paths))


if __name__ == "__main__":
    unittest.main()
