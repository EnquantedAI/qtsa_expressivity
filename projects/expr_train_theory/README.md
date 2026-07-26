# Expressivity and trainability notes

This directory contains my current work on the mathematical side of the project. At the moment it focuses on the pure-state Quantum Fisher Information Matrix (QFIM).

The QFIM code is written in NumPy and does not depend on a particular quantum framework. I used small examples with known answers to check the implementation before connecting it to PennyLane. The PennyLane adapter uses the feature map from `src/models.py` and returns the full state of the circuit, which is needed for the QFIM calculation.

## Contents

- `NOTES.md` — short notes on the notation, the role of the different metrics and the assumptions used in the experiments;
- `qfim/` — QFIM implementation, validation examples, tests and a small width/depth experiment;
- `CHANGELOG.md` — brief summary of the work done in this directory.

## Running the tests

From the repository root:

```bash
python -m unittest discover \
  -s projects/expr_train_theory/qfim/tests \
  -t . \
  -v
```

The NumPy tests can be run without PennyLane. Tests which use the shared circuit are skipped when PennyLane is not installed.

To check the analytical examples:

```bash
python -m projects.expr_train_theory.qfim.run_validation
```

With the project dependencies installed, the PennyLane check can be run with:

```bash
python -m projects.expr_train_theory.qfim.run_pennylane_validation
```

A minimal architecture sweep can be started with:

```bash
python -m projects.expr_train_theory.qfim.experiments.run_architecture_sweep \
  --widths 2 \
  --depths 1 \
  --samples 1
```

The PennyLane experiment has not yet been run on the full project environment, so there are no QFIM results for the shared QNN in the repository yet.
