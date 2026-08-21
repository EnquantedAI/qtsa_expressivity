# Perturbation checks

A small check of how stable trajectory participation is under nearby trajectories.

There are two variants:

- perturb the state snapshots directly;
- perturb the variational parameters and rebuild the layered trajectory.

For each perturbation size the script records $d_{TP}$, its change from the baseline, numerical rank, entropy and the mean snapshot infidelity.

```bash
python -m projects.expr_train_theory.trajectory_participation.perturbation_stability.run_study
```

The point here is just to check continuity and numerical sensitivity before using the metric in larger architecture comparisons.
