# Expressivity and trainability

This folder contains my work on the mathematical side of the project. The code is split into four main areas:

- `qfim/` — a small pure-state QFIM implementation, analytical checks and a PennyLane adapter;
- `effective_dimension_checks/` — simple CFIM/QFIM examples used while reviewing the effective-dimension code;
- `krylov/` — Lanczos/Arnoldi routines and small experiments for checking possible Krylov-based diagnostics;
- `trajectory_participation/` — a reference implementation of the layer-state participation idea based on the singular-value spectrum of a trajectory.


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

For the small trajectory/Fisher comparison:

```bash
python -m projects.expr_train_theory.trajectory_participation.cross_metric_validation.run_study
```

For a compact side-by-side check of dTP, QFIM, CFIM and QNTK on the same small models:

```bash
python -m projects.expr_train_theory.trajectory_participation.metric_profile.run_profile
```

For the small agreement/disagreement report across the reference metrics:

```bash
python -m projects.expr_train_theory.trajectory_participation.metric_agreement.run_report
```

For the overlap/frame-potential identity behind trajectory participation:

```bash
python -m projects.expr_train_theory.trajectory_participation.overlap_view.run_check
```

For the matched entanglement-graph work, the stable reproducible entry point is:

```bash
python -m projects.expr_train_theory.trajectory_participation.graph_topology.run_reproducible_study
```

Use `--show-preset` to inspect the frozen `topology_robust_v1` configuration without running it. The focused topology runners are development/diagnostic tools and are documented separately in `trajectory_participation/graph_topology/README.md`.
