# dTP calibration

A few reference quantities for interpreting trajectory participation values.

For a trajectory with $m$ snapshots in a Hilbert space of dimension $D$,

$$
1 \le d_{TP} \le \mathrm{rank}(\Psi) \le \min(m,D).
$$

Besides raw $d_{TP}$, the helper reports $d_{TP}/\mathrm{rank}(\Psi)$ and $d_{TP}/\min(m,D)$. The first says how evenly the trajectory weight is spread across the directions it actually uses; the second also accounts for unused directions that were available in principle.

It also reports the entropy effective dimension $\exp(S)$ and the stable rank $1/p_{\max}$. These are useful checks because they use the same trajectory spectrum but emphasize it differently.

```bash
python -m projects.expr_train_theory.trajectory_participation.calibration.run_calibration
```
