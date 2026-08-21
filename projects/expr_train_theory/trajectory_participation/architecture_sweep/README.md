# Small architecture sweep

This runs the trajectory participation calculation on the current shared-QNN layout while changing the number of layers, qubits, feature map and reuploading choice.

The input window length is kept separate from the qubit count, so the sweep also covers the current model behaviour with fewer or more qubits than input values.

For each random initialisation the script stores $d_{TP}$, a normalized value, numerical rank and trajectory entropy. The raw data and a small grouped summary are written to `results/`.

```bash
python -m projects.expr_train_theory.trajectory_participation.architecture_sweep.run_sweep
```

A smaller run can be useful while checking changes:

```bash
python -m projects.expr_train_theory.trajectory_participation.architecture_sweep.run_sweep \
  --layers 1 2 --qubits 1 2 --feature-maps Y --reupload none Y --samples 2
```
