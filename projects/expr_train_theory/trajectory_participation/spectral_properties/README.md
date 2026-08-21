# Spectral properties of trajectory participation

This folder keeps the spectral interpretation of trajectory participation in one place.
If $p_i$ are the normalized squared singular values of the trajectory matrix, then

$$
d_{TP} = \frac{1}{\sum_i p_i^2} = \exp(H_2(p)),
$$

so trajectory participation is the Renyi-2 effective dimension of the trajectory spectrum.
The same spectrum also gives the Shannon effective dimension

$$
d_1 = \exp(H_1(p))
$$

and the stable-rank version

$$
d_\infty = \frac{1}{\max_i p_i}.
$$

For a finite spectrum these quantities satisfy

$$
d_\infty \le d_{TP} \le d_1 \le \mathrm{rank}(\Psi).
$$

The inequalities are useful when interpreting $d_{TP}$: it is a soft dimension between the most conservative largest-weight estimate and the Shannon effective rank, rather than a hard count of non-zero directions.

Run the small check with

```bash
python -m projects.expr_train_theory.trajectory_participation.spectral_properties.run_check
```
