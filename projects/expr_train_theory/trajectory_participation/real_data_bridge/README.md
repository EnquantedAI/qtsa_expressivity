# Real-data trajectory bridge

This module runs the native shared-QNN trajectory diagnostics on the preprocessed
NARMA10 and Mackey-Glass windows already stored by
`projects/trainability_effective_dim`.

The bridge is deliberately read-only: it does not regenerate, rescale, copy, or
modify the other project's datasets. By default it uses deterministic, evenly
spaced windows from the held-out `test` split rather than selecting examples by
observed metric values.

For every selected window and shared-QNN architecture it records equal-weight
and Fubini-Study-weighted $d_{TP}$, depth-resolved trajectory growth, and the
ancilla entropy/purity diagnostics when ancilla wires are present. The stored
forecast target is kept only as provenance; it is not used to compute any of
these state-space quantities.

The default study uses random **untrained** weights. Its purpose is therefore to
compare how real project inputs probe the circuit geometry before training. It
must not be interpreted as forecasting performance or as a trained-model
result.

Run from the repository root with:

```bash
python -m projects.expr_train_theory.trajectory_participation.real_data_bridge.run_study
```

The runner writes raw observations, depth-resolved rows, summaries, and a JSON
manifest to `results/`. The manifest records the exact held-out window indices,
architecture grid, seed, and whether targets or trained weights were used.

For repeated-seed studies the window-selection seed can be fixed independently of the weight seed, so the exact same held-out inputs can be reused across parameter initialisations.
