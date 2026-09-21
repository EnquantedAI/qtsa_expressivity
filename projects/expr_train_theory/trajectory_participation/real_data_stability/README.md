# Real-data stability study

This module asks whether the architecture effects seen by the native shared-QNN
trajectory diagnostics are stable across two independent nuisance axes: held-out
time-series windows and random parameter initialisations.

It reuses the read-only NARMA10 and Mackey-Glass bridge. The same held-out window
indices are fixed across all repeat seeds. Architecture differences are computed
**before averaging**, within the same dataset window and the same random parameter
draw.

Two matched effect families are reported:

- ancilla effects: compare additional ancilla qubits with the zero-ancilla
  baseline at fixed depth and encoding;
- depth effects: compare deeper circuits with the shallowest configured depth at
  fixed ancilla count and encoding.

The main diagnostics are normalized equal-weight and Fubini--Study-weighted
$d_{TP}$, their raw counterparts, projective path length, and ancilla-sector
quantities. The summary reports sign fractions, variability of window-conditional
and weight-draw-conditional means, and a crossed bootstrap confidence interval
that resamples both held-out windows and parameter draws.

`stable_positive` or `stable_negative` means that the bootstrap interval for the
matched mean effect stays on one side of zero. `unresolved` means that it does
not. These labels are finite-sample diagnostics, not universal statements about
the architecture family.

The default weights are still random and untrained. The study therefore tests
stability of state-space/entanglement diagnostics, not forecasting performance.

Run from the repository root with:

```bash
python -m projects.expr_train_theory.trajectory_participation.real_data_stability.run_study
```
