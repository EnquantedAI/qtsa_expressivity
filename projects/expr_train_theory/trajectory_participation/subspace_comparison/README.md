# Hard subspace dimension vs trajectory participation

Gram-Schmidt and SVD rank answer a different question from $d_{TP}$.

For a trajectory matrix $\Psi$, the hard dimension is

$$
r = \mathrm{rank}(\Psi).
$$

It only asks how many linearly independent directions occur. In contrast,

$$
d_{TP}=\frac{1}{\sum_i p_i^2}
$$

also uses the distribution of spectral weight between those directions. Two trajectories can therefore span exactly the same subspace and still have different $d_{TP}$.

A simple example is

$$
\{|0\rangle,|1\rangle\}
$$

versus

$$
\{|0\rangle,|0\rangle,|0\rangle,|1\rangle\}.
$$

Both have rank 2. The first has $d_{TP}=2$, while the second has

$$
d_{TP}=\frac{1}{(3/4)^2+(1/4)^2}=\frac{8}{5}.
$$

This is useful for the trajectory idea because Gram-Schmidt gives the hard accessible subspace, while $d_{TP}$ tells us how evenly the sampled trajectory occupies it. The helper also reports $d_{TP}/r$ as a simple utilization ratio.

Run the small check with:

```bash
python -m projects.expr_train_theory.trajectory_participation.subspace_comparison.run_check
```
