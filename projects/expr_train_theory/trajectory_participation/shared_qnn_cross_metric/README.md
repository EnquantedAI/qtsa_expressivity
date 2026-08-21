# Shared-QNN cross-metric sweep

Small sweep comparing trajectory participation with Fisher-based diagnostics on the same QNN instances.

For each sampled circuit the script uses the layer snapshots for $d_{TP}$ and the final state for QFIM and computational-basis CFIM. For the CFIM, the generic output variable $y$ in $p_\theta(y\mid x)$ is the computational-basis outcome $z$, i.e. the implementation uses $p_\theta(z\mid x)$. The point is to keep the architecture, input and parameters fixed while changing only the metric.

Run a small study with:

```bash
python -m projects.expr_train_theory.trajectory_participation.shared_qnn_cross_metric.run_study
```

The script writes raw rows, per-architecture summaries and a small correlation table to `results/`.
