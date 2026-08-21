# Validation suite

This pulls together the small checks that were added while working on trajectory participation.

The boundary and zero-perturbation cases are treated as hard checks because the expected answer is known. Depth, reuploading and sampling-density trends are only reported as diagnostics; they are not assumed to be monotone.

Run from the repository root:

```bash
python -m projects.expr_train_theory.trajectory_participation.validation_suite.run_validation
```

The runner writes a short Markdown summary and the full JSON report to `results/`.
