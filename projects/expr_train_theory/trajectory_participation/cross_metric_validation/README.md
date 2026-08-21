# Cross-metric checks

A few small cases comparing trajectory participation with Fisher-based diagnostics on the same parametric model.

The point is not to combine the metrics into one score. $d_{TP}$ uses the sequence of states visited by the circuit, while QFIM/CFIM describe local sensitivity of the final model state or measurement distribution to its parameters. These can disagree for good reasons.

Current reference cases include repeated $R_Y$ layers, a phase-only $R_Z$ trajectory, and a mixed-axis one-qubit trajectory.

Run from the repository root:

```bash
python -m projects.expr_train_theory.trajectory_participation.cross_metric_validation.run_study
```

The script writes a small CSV and metadata file to `results/`.
