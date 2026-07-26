# QFIM width/depth sweep

This script evaluates QFIM diagnostics for a small grid of circuit widths and depths. It uses the shared feature map and `StronglyEntanglingLayers` through the PennyLane adapter.

Example:

```bash
python -m projects.expr_train_theory.qfim.experiments.run_architecture_sweep \
  --widths 2 3 4 \
  --depths 1 2 3 \
  --samples 3
```

The script writes:

- one CSV file with all sampled circuits;
- one CSV file with values grouped by architecture;
- one JSON file with the experiment settings.

The default output directory is `results/`. Generated CSV and JSON files are ignored by Git in this project directory.
