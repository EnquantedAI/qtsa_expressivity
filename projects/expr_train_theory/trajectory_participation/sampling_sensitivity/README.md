# Sampling sensitivity

This check looks at something that matters when comparing circuits with different numbers of layers: $d_{TP}$ is computed from a finite list of snapshots, so changing the sampling density can change the value even if the underlying path is the same.

Two simple checks are included:

- resampling the same one-qubit arc with different numbers of snapshots;
- duplicating one state many times.

The second example also shows why snapshot multiplicity matters. With equal snapshot weights, repeated states receive more total weight. A weighted version of $d_{TP}$ is included so that this can be controlled when needed.

For snapshots with weights $w_k$ I use

$$
\rho_T = \sum_k w_k |\psi_k\rangle\langle\psi_k|,
\qquad
\sum_k w_k = 1,
$$

and

$$
d_{TP}^{(w)} = \frac{1}{\mathrm{Tr}(\rho_T^2)}.
$$

For a uniformly sampled trajectory this reduces to the current equal-weight definition.

```bash
python -m projects.expr_train_theory.trajectory_participation.sampling_sensitivity.run_study
```
