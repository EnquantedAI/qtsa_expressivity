# dTP / QNTK architecture check

This is a small NumPy sweep for comparing trajectory participation with QNTK diagnostics on the same layered circuits.

The sweep varies depth, qubit count, reuploading and whether the toy circuit contains entangling gates. For every sampled circuit it uses one small input set, averages normalized $d_{TP}$ across the inputs and builds the QNTK from the final $\langle Z_0\rangle$ output.

The script records QNTK rank, trace, effective rank and element variance together with the trajectory quantities. The correlations are exploratory; they are not used as a definition of expressivity or trainability.

```bash
python -m projects.expr_train_theory.trajectory_participation.qntk_architecture_sweep.run_study
```

Generated CSV/JSON files are written to `results/`.
