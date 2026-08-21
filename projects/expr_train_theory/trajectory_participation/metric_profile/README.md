# Metric profile

Small reference cases for looking at trajectory participation, QFIM, computational-basis CFIM and QNTK side by side.

The point is not to combine them into one score. The cases are chosen so that some parameter directions are visible to one metric and invisible to another. This makes it easier to check what each quantity is actually responding to.

Run from the repository root:

```bash
python -m projects.expr_train_theory.trajectory_participation.metric_profile.run_profile
```

Use `--save` to write the CSV and JSON summary to the local `results/` directory.
