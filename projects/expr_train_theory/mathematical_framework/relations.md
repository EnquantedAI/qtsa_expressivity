# Relations and interpretation

The project metrics are intentionally complementary.  They should be compared
on matched circuits and data, but agreement is not a correctness condition.

## Trajectory participation identities

For normalized equal-weight snapshots,

```math
\rho_T=\frac1m\sum_k|\psi_k\rangle\langle\psi_k|,
```

and

```math
d_{TP}
=
\frac{1}{\mathrm{Tr}(\rho_T^2)}
=
\frac{m^2}{\sum_{k,\ell}|\langle\psi_k|\psi_\ell\rangle|^2}.
```

If $p_i$ is the normalized trajectory spectrum, then

```math
d_{TP}=\exp(H_2(p)),
\qquad
H_2(p)=-\log\sum_i p_i^2.
```

The current spectral checks use the ordering

```math
d_\infty\le d_{TP}\le d_1\le \mathrm{rank}(\Psi),
```

where $d_\infty$ is the stable-rank view and
$d_1=\exp(H_1)$ is the Shannon effective dimension of the same spectrum.
Thus hard rank answers how many independent directions are reached, while
$d_{TP}$ also measures how evenly the trajectory occupies them.

### Invariances

Equal-weight $d_{TP}$ is unchanged by:

- a common unitary applied to every snapshot;
- independent global phases on the snapshots;
- a permutation of the snapshot list.

The FS-weighted version is also projective/global-phase invariant, but it uses
the ordering of the trajectory because consecutive arc lengths define the
weights.

Duplicating snapshots can change equal-weight $d_{TP}$ because it changes the
empirical trajectory mixture.  This is the motivation for reporting the
Fubini--Study-weighted diagnostic alongside it when snapshot density changes.

## QFIM versus CFIM

QFIM is a local state-space quantity.  It responds to parameter directions
that change the physical pure state.

CFIM is defined only after a measurement distribution has been chosen.  The
current Effective Dimension implementation uses computational-basis
probabilities $p_\theta(z\mid x)$.  A parameter direction can therefore have
non-zero QFIM while having zero computational-basis CFIM; the phase-only
reference examples in this project are designed to show exactly this case.

Consequently, CFIM disagreement with QFIM is not by itself an implementation
failure.  The measurement model must be part of the interpretation.

## Effective Dimension versus Fisher rank

GED and LED are not simply ranks of a single Fisher matrix.  They aggregate a
nonlinear log-determinant functional over a distribution of parameter-space
Fisher matrices and depend additionally on the theoretical dataset size $n$.

The current conventions therefore require the following metadata to accompany
a result:

- measurement used to define CFIM;
- input sample set used in the empirical Fisher average;
- global or local parameter sampler;
- $N_\theta$;
- for LED, the radius $\epsilon$;
- theoretical dataset-size values $n$;
- whether the returned value is divided by parameter count $d$;
- probability cutoff used in the CFIM calculation.

Without these choices, two Effective Dimension values need not be directly
comparable.

## QNTK versus state-space quantities

QNTK is attached to a selected output function $f_\theta(x)$, not directly to
the full quantum state.  A parameter may change $|\psi_\theta(x)\rangle$ and
therefore affect QFIM or trajectory participation while leaving a particular
observable $f_\theta(x)$ unchanged.  Such a direction is invisible to the
QNTK built from that output.

This distinction is useful rather than problematic:

- $d_{TP}$ probes the set of states visited across circuit depth;
- QFIM probes local state sensitivity to parameters;
- CFIM probes local sensitivity visible through a chosen measurement;
- GED/LED aggregate that measurement-level sensitivity over parameter regions;
- QNTK probes sensitivity of the chosen model output over the dataset.

## Ancilla diagnostics

For a bipartition into system and ancilla, let

```math
\rho_{anc}=\mathrm{Tr}_{sys}|\psi\rangle\langle\psi|.
```

The current trajectory/ancilla study uses

```math
S(\rho_{anc})=-\mathrm{Tr}(\rho_{anc}\log\rho_{anc}),
\qquad
P_{anc}=\mathrm{Tr}(\rho_{anc}^2),
\qquad
d_{anc}=P_{anc}^{-1}.
```

Although both $d_{anc}$ and $d_{TP}$ are inverse-purity quantities, the density
matrices are different:

- $\rho_{anc}$ describes a subsystem of one instantaneous pure state;
- $\rho_T$ describes a mixture over different trajectory snapshots.

A correlation between them across depth is therefore descriptive and should
not be interpreted as equality or causation.

## Cross-metric comparison rules

For presentation and publication-oriented experiments, comparisons should be
matched whenever possible: same data, same parameter draw, same architecture
except for the controlled factor being changed.

Architecture trends such as increasing depth, width, ancilla count or graph
density are empirical diagnostics.  The framework does not assume that any of
$d_{TP}$, QFIM, CFIM, GED/LED or QNTK must vary monotonically with those
choices.

When a metric depends on an additional convention, report that convention
explicitly rather than hiding it in implementation defaults.  This is
especially important for snapshot weighting, measurements, output observables,
parameter-space sampling and Effective Dimension normalization.
