# dTP / QNTK cross-check

Small reference comparison between trajectory participation and a QNTK built from the Jacobian of model outputs over a dataset.

For a Jacobian $J$ with samples in rows and parameters in columns, the kernel used here is

$$
K = J J^T.
$$

The code keeps this independent of the implementation in `projects/expressivity_krylov/`. The point is to have a tiny NumPy reference for boundary tests and later comparison with the project QNTK pipeline.

The toy circuit has one encoded input followed by an $R_Y$ and an $R_Z$ layer. Its output is a $Z$ expectation value. This gives a useful case where the phase parameter changes the state trajectory (and therefore can change $d_{TP}$) while remaining invisible to this particular scalar output and its QNTK.

```bash
python -m projects.expr_train_theory.trajectory_participation.qntk_cross_metric.run_study
```
