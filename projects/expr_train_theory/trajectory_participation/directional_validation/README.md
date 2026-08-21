# Directional checks

A small check of how the current trajectory metric reacts when the reference layered circuit is made deeper or when data reuploading is switched on.

This is not treated as a proof that $d_{TP}$ has to be monotone. The script records the trend, the number of upward/downward steps and a Spearman correlation instead of assuming a direction in advance.

The reuploading comparison is paired: the same input and variational weights are used with and without the extra encoding block.

```bash
python -m projects.expr_train_theory.trajectory_participation.directional_validation.run_study
```

Results are written to `results/` as raw values, depth summaries and paired reuploading differences.
