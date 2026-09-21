# Trajectory / ancilla coupling diagnostic

This study connects the native shared-QNN trajectory diagnostics with the ancilla-sector diagnostics used in the current PennyLane model.  States are collected from `src.models.gqnn(..., add_snaps=True)` through the native snapshot adapter.

For the partial trajectory through depth $l$ it records equal-weight and Fubini--Study-weighted $d_{TP}$.  At the same depth it traces out the system wires and evaluates the reduced ancilla state

```math
\rho_{\mathrm{anc}}^{(l)}=\mathrm{Tr}_{\mathrm{sys}}\left(|\psi_l\rangle\langle\psi_l|\right).
```

The ancilla diagnostics are von Neumann entropy, purity and

```math
d_{\mathrm{anc}}^{(l)}=\frac{1}{\mathrm{Tr}[(\rho_{\mathrm{anc}}^{(l)})^2]}.
```

The last quantity is a Renyi-2 effective Schmidt dimension.  It has the same inverse-purity form as trajectory participation, but the density matrices are different: $d_{TP}$ describes occupation of the trajectory mixture, while $d_{\mathrm{anc}}$ describes instantaneous system--ancilla entanglement for a globally pure snapshot.  The two quantities should therefore not be identified.

The study saves depth-resolved rows together with within-trajectory Pearson diagnostics between $d_{TP}$ growth and ancilla entropy/effective dimension.  It also reports correlations of successive increments when enough depths are available.  These are descriptive associations only; depth is a common ordering variable and the correlations are not evidence of causality.

The zero-ancilla case is kept as an exact reference: entropy and mixedness are zero, purity and $d_{\mathrm{anc}}$ are one, and correlation fields involving the constant ancilla series are left undefined.

The default runner uses synthetic inputs and untrained matched parameter prefixes, so it is a pipeline/architecture diagnostic rather than a trained-model result.

```bash
python -m projects.expr_train_theory.trajectory_participation.ancilla_coupling.run_study
```
