# Arc-length weighting

Equal snapshot weights make trajectory participation depend partly on how often a path is sampled. This helper uses the projective distance between consecutive pure states as the trajectory coordinate instead.

For normalized states,

$$
d_{FS}(|\psi\rangle,|\phi\rangle)
= \arccos |\langle\psi|\phi\rangle|.
$$

The cumulative sum of consecutive distances gives a discrete projective arc length. Standard trapezoidal weights on that coordinate are then used in

$$
\rho_T = \sum_k w_k |\psi_k\rangle\langle\psi_k|,
\qquad
d_{TP}^{(FS)} = \frac{1}{\mathrm{Tr}(\rho_T^2)}.
$$

This does not make trajectories from different circuit depths identical objects. An extra layer can genuinely change the path. The point is narrower: dense sampling of a slowly moving part of one path should not automatically give that region more weight just because more snapshots were recorded there.

The Fubini--Study distance is phase invariant, so independent global phases on the snapshots do not change the weights.

```bash
python -m projects.expr_train_theory.trajectory_participation.arc_length_weighting.run_check
```
