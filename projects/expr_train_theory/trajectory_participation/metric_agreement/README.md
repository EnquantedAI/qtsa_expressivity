# Metric agreement check

A small follow-up to `metric_profile/`. It turns the reference cases into simple visibility patterns so it is easier to see where the metrics agree and where they are looking at different things.

For now a metric is treated as "active" when it sees a non-trivial direction in the corresponding case. This is only a diagnostic for the toy examples, not a general score for model quality.

Run from the repository root:

```bash
python -m projects.expr_train_theory.trajectory_participation.metric_agreement.run_report
```

Use `--save` to write the CSV and JSON summary to `results/`.
