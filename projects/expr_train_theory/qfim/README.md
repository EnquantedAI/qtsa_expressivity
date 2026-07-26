# QFIM code

`core.py` contains a NumPy implementation of the pure-state Quantum Fisher Information Matrix. It accepts a function which maps a parameter vector to a normalized state vector.

The derivatives are calculated with central finite differences. This is meant as a simple reference implementation rather than the fastest version.

## Files

- `core.py` — QFIM calculation and numerical diagnostics;
- `validation_models.py` — small states with known QFIM values;
- `run_validation.py` — prints the analytical examples;
- `pennylane_adapter.py` — builds a state-returning version of the shared circuit;
- `run_pennylane_validation.py` — small PennyLane check;
- `experiments/` — width/depth sweep;
- `tests/` — unit tests.

Run the tests from the repository root:

```bash
python -m unittest discover \
  -s projects/expr_train_theory/qfim/tests \
  -t . \
  -v
```
