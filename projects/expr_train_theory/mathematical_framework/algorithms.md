# Abstract algorithms

The algorithms below define mathematical input/output contracts.  They do not
assume NumPy, PyTorch, PennyLane or a particular differentiation backend.

## 1. Equal-weight trajectory participation dimension

**Input:** normalized snapshots $|\psi_0\rangle,\ldots,|\psi_L\rangle$.

1. Set $m=L+1$ and $w_k=1/m$.
2. Form
   ```math
   \rho_T=\sum_{k=0}^{L}w_k|\psi_k\rangle\langle\psi_k|.
   ```
3. Compute the purity
   ```math
   P_T=\mathrm{Tr}(\rho_T^2).
   ```
4. Return
   ```math
   d_{TP}=P_T^{-1}.
   ```

Equivalent Gram form:

```math
d_{TP}
=
\frac{m^2}
{\sum_{k,\ell}|\langle\psi_k|\psi_\ell\rangle|^2}
=
\frac{(\mathrm{Tr}\,G)^2}{\mathrm{Tr}(G^2)}.
```

Equivalent spectral form: if $p_i$ are the normalized squared singular values
of $\Psi$, then

```math
d_{TP}=\frac{1}{\sum_i p_i^2}.
```

## 2. Fubini--Study-weighted trajectory participation

**Input:** the same ordered trajectory.

1. Compute consecutive projective distances
   ```math
   \delta_k=\arccos(|\langle\psi_{k-1}|\psi_k\rangle|).
   ```
2. Use cumulative Fubini--Study arc length as a one-dimensional quadrature
   coordinate.
3. Assign normalized trapezoidal weights $w_k$ to the snapshots.
4. Form
   ```math
   \rho_T^{(FS)}=\sum_k w_k|\psi_k\rangle\langle\psi_k|.
   ```
5. Return
   ```math
   d_{TP}^{(FS)}
   =
   \frac{1}{\mathrm{Tr}[(\rho_T^{(FS)})^2]}.
   ```

If all consecutive projective distances vanish, use equal weights.  This
weighted quantity is a sampling diagnostic; it is not assumed to replace the
equal-weight definition in every experiment.

## 3. Pure-state quantum Fisher information matrix

**Input:** a differentiable normalized state map
$\theta\mapsto|\psi_\theta(x)\rangle$ for fixed $x$.

For every pair of parameters $(i,j)$ compute

```math
F^Q_{ij}(\theta,x)
=
4\,\mathrm{Re}\left[
\langle\partial_i\psi|\partial_j\psi\rangle
-
\langle\partial_i\psi|\psi\rangle
\langle\psi|\partial_j\psi\rangle
\right].
```

The derivatives may be obtained analytically, by automatic differentiation,
parameter-shift rules, or by a numerical reference method.  The formula, not
the differentiation backend, defines the object.

Useful diagnostics are the numerical rank, relative rank, trace and spectrum.

## 4. Classical Fisher information matrix

**Input:** a differentiable conditional distribution
$p_\theta(y\mid x)$ for a specified measurement model.

For fixed $x$ compute

```math
F^C_{ij}(\theta,x)
=
\sum_{y\in\mathcal Y}
\frac{
\partial_i p_\theta(y\mid x)\,
\partial_j p_\theta(y\mid x)
}{p_\theta(y\mid x)}.
```

Numerically negligible probabilities may be excluded using an explicit cutoff.
The measurement outcome space must be reported with the result.  The current
shared Effective Dimension code uses computational-basis outcomes $y=z$.

For an input set $X=(x_1,\ldots,x_N)$ define the input-averaged empirical
Fisher at fixed parameters by

```math
F_{\mathrm{emp}}(\theta)
=
\frac{1}{N}\sum_{a=1}^{N}F^C(\theta,x_a).
```

## 5. Global and local Effective Dimension

**Input:** an input set $X$, a parameter-space distribution $\pi$, a number of
parameter samples $N_\theta$, and theoretical dataset sizes $n$.

1. Draw
   ```math
   \theta_k\sim\pi,
   \qquad k=1,\ldots,N_\theta.
   ```
2. Compute $F_{\mathrm{emp}}(\theta_k)$ for every draw.
3. Let $d$ be the number of scalar trainable parameters and normalize
   according to the current repository convention:
   ```math
   \widehat F(\theta_k)
   =
   \frac{
   d\,F_{\mathrm{emp}}(\theta_k)
   }{
   \mathrm{Tr}\left[
   \frac{1}{N_\theta}\sum_r F_{\mathrm{emp}}(\theta_r)
   \right]
   }.
   ```
4. For each theoretical dataset size $n$, set
   ```math
   \kappa(n)=\frac{n}{2\pi\log n}.
   ```
5. For $\kappa>1$, return
   ```math
   d_{\mathrm{eff}}(n)
   =
   \frac{2}{\log\kappa}
   \log\left[
   \frac{1}{N_\theta}
   \sum_{k=1}^{N_\theta}
   \sqrt{\det(I+\kappa\widehat F(\theta_k))}
   \right].
   ```

The distinction between GED and LED is the parameter distribution $\pi$:

- **GED:** global uniform sampling over the selected parameter domain;
- **LED:** uniform sampling in an $\epsilon$-ball around the current parameter
  vector.

When comparing circuits with different parameter counts, the repository may
also report $d_{\mathrm{eff}}/d$.  This normalization of the final value is
separate from the Fisher normalization above.

## 6. Quantum neural tangent kernel

**Input:** a selected scalar output $f_\theta(x)$ and dataset
$X=(x_1,\ldots,x_N)$.

1. Form the Jacobian
   ```math
   J_{a i}=\partial_{\theta_i}f_\theta(x_a).
   ```
2. Form the kernel
   ```math
   K=JJ^\top,
   \qquad
   K_{ab}=\nabla_\theta f_\theta(x_a)\cdot
   \nabla_\theta f_\theta(x_b).
   ```
3. Analyse the positive spectrum of $K$ using quantities such as rank, trace,
   condition number and participation/effective rank.

The current small reference uses

```math
d_{\mathrm{eff}}(K)
=
\frac{1}{\sum_r q_r^2},
\qquad
q_r=\frac{\lambda_r(K)}{\mathrm{Tr}\,K},
```

when $\mathrm{Tr}\,K>0$.

## 7. Native trajectory acquisition from the shared QNN

For an integration experiment the mathematical trajectory must be obtained
from the canonical shared circuit rather than reconstructed independently.
The backend contract is:

1. evaluate the shared QNN for $(x,\theta)$ with layer snapshots enabled;
2. extract the ordered states `depth_0`, ..., `depth_L`;
3. verify normalization and the expected Hilbert-space dimension;
4. pass the resulting state sequence to the backend-independent trajectory
   algorithms above.

The independently reconstructed NumPy/PennyLane path remains useful as a
reference for parity tests, but it is not the source of states for a native
shared-model result.
