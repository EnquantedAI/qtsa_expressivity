# QFIM architecture sweep

Small width/depth sweep for the shared PennyLane circuit. Raw values, summaries and settings are written to `results/`.

```bash
python -m projects.expr_train_theory.qfim.experiments.run_architecture_sweep --widths 2 3 --depths 1 2 --samples 3
```
