# Analytic boundary checks

A few small cases where the expected answer is known before running the code.

The point is to catch basic mistakes before using the metrics in larger sweeps. The checks cover collapsed and orthogonal trajectories, a constant-output QNTK, and redundant parameter directions.

```bash
python -m projects.expr_train_theory.trajectory_participation.boundary_validation.run_validation
```
