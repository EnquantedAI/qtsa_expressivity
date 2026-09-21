# Mathematical framework

This folder collects the backend-independent mathematical objects used in the
expressivity/trainability work.  The aim is to keep the definitions separate
from any particular NumPy, PyTorch or PennyLane implementation.

The common setup is a parametrized quantum model

```math
(x,\theta) \longmapsto |\psi_\theta(x)\rangle,
```

possibly together with an ordered sequence of intermediate states

```math
\mathcal T_\theta(x)
=
\bigl(|\psi_0\rangle,\ldots,|\psi_L\rangle\bigr),
```

a measurement model $p_\theta(y\mid x)$, and/or a classical model output
$f_\theta(x)$.  The metrics in the project probe different objects derived
from this common model:

| Quantity | Primary object | Main question |
| --- | --- | --- |
| $d_{TP}$ | ordered state snapshots | how broadly the recorded trajectory occupies state space |
| FS-weighted $d_{TP}$ | projective trajectory + arc-length weights | how broadly the path is occupied after reducing snapshot-density bias |
| QFIM | local derivatives of the quantum state | which parameter directions change the physical state |
| CFIM | derivatives of a chosen measurement distribution | which parameter directions are visible to that measurement |
| GED / LED | a parameter-space family of empirical CFIMs | how many parameter directions are statistically distinguishable globally / locally |
| QNTK | Jacobian of the chosen model output on a dataset | how parameter changes couple to the output function over the data |

These quantities are not expected to agree or to be monotone with circuit
depth.  Agreement and disagreement are themselves diagnostics because the
metrics live on different derived objects.

## Files

- `notation.md` fixes symbols and conventions used across the project;
- `algorithms.md` gives backend-independent abstract algorithms;
- `relations.md` records the main identities, invariances and distinctions that
  are useful when interpreting cross-metric experiments.

## Implementation map

The framework deliberately does not own a numerical backend.  Current
implementations are mapped as follows:

- trajectory participation: `trajectory_participation/core.py`;
- Fubini--Study weighting: `trajectory_participation/arc_length_weighting/`;
- native shared-QNN snapshots: `trajectory_participation/shared_qnn.py`;
- pure-state QFIM reference: `qfim/core.py`;
- computational-basis CFIM and GED/LED: read-only shared implementation in
  `projects/trainability_effective_dim/core/measures/`;
- small QNTK reference: `trajectory_participation/qntk_cross_metric/qntk.py`.

The NumPy routines in `expr_train_theory` are reference/validation
implementations.  The shared experimental circuit is the PennyLane model in
`src.models`; trajectory studies should obtain its snapshots through the
native snapshot path rather than reconstructing the shared model when an
integration result is required.

## Scope

This document fixes the conventions currently used by the repository.  It is
not meant to assert that every metric is unique or canonical in the broader
literature.  In particular, CFIM depends on the chosen measurement, GED/LED
depend on parameter-space and dataset-size conventions, and QNTK depends on
the chosen output function.
