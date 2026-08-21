# Metric uncertainty

The earlier cross-metric sweeps report point estimates of correlations. This folder adds a small bootstrap check so we can see how stable those numbers are across parameter samples.

For now it reports percentile intervals for Pearson and Spearman correlations between selected dTP and QNTK quantities. The main purpose is to avoid reading too much into a correlation from a small exploratory sweep.

Run from the repository root:

```bash
python -m projects.expr_train_theory.trajectory_participation.metric_uncertainty.run_bootstrap
```

Add `--save` to write the table and run metadata to `results/`.
