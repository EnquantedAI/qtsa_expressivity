# Expressivity and trainability

This folder contains my work on the mathematical side of the project. The code is split into three parts:

- `qfim/` — a small pure-state QFIM implementation, analytical checks and a PennyLane adapter;
- `effective_dimension_checks/` — simple CFIM/QFIM examples used while reviewing the effective-dimension code;
- `krylov/` — Lanczos/Arnoldi routines and a set of small experiments for checking possible Krylov-based diagnostics.


## Tests

From the repository root:

```bash
python -m unittest discover -s projects/expr_train_theory -t . -v
```

A few useful entry points:

```bash
python -m projects.expr_train_theory.qfim.run_validation
python -m projects.expr_train_theory.effective_dimension_checks.run_examples
python -m projects.expr_train_theory.krylov.run_validation
```

The larger scripts under `qfim/experiments/` and `krylov/*/` write their output to local `results/` folders. Generated results are not committed.
