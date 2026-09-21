# Notation

## Model, data and parameters

Let

```math
x\in\mathcal X,
\qquad
\theta=(\theta_1,\ldots,\theta_d)\in\Theta\subseteq\mathbb R^d.
```

The shared parametrized quantum circuit defines a normalized pure state

```math
|\psi_\theta(x)\rangle\in\mathcal H,
\qquad
\dim\mathcal H=2^q,
```

where $q$ is the number of qubits.  Global phase is physically irrelevant, so
state-space comparisons should depend on the projective state rather than on a
particular vector representative whenever possible.

For a dataset write

```math
X=(x_1,\ldots,x_N).
```

The symbol $d$ always denotes the number of scalar trainable parameters.  It
must not be confused with a trajectory participation dimension or an
input-space dimension.

## Layer trajectory

For a layered circuit we record an ordered sequence

```math
\mathcal T_\theta(x)
=
\bigl(|\psi_0\rangle,|\psi_1\rangle,\ldots,|\psi_L\rangle\bigr),
```

where $|\psi_0\rangle$ is the state after the initial feature encoding and
$|\psi_\ell\rangle$ is the state after variational layer $\ell$.  There are

```math
m=L+1
```

snapshots.  In the current shared PennyLane model these correspond to the
named snapshots `depth_0`, ..., `depth_L`.

The trajectory matrix is

```math
\Psi
=
\begin{bmatrix}
|\psi_0\rangle & \cdots & |\psi_L\rangle
\end{bmatrix},
```

and its snapshot Gram matrix is

```math
G=\Psi^\dagger\Psi,
\qquad
G_{k\ell}=\langle\psi_k|\psi_\ell\rangle.
```

All snapshots are assumed normalized before the trajectory metrics are
constructed.

## Trajectory weights

A set of non-negative snapshot weights is denoted

```math
w=(w_0,\ldots,w_L),
\qquad
w_k\ge 0,
\qquad
\sum_k w_k=1.
```

The associated trajectory density matrix is

```math
\rho_T
=
\sum_{k=0}^{L} w_k |\psi_k\rangle\langle\psi_k|.
```

Equal weighting means $w_k=1/m$.

For Fubini--Study weighting define consecutive projective distances

```math
\delta_k
=
d_{FS}(\psi_{k-1},\psi_k)
=
\arccos\!\left(|\langle\psi_{k-1}|\psi_k\rangle|\right),
\qquad k=1,\ldots,L.
```

The current convention uses trapezoidal quadrature weights along the
cumulative projective arc length.  If the total path length is zero, it falls
back to equal weights.

## Measurement distribution

For a measurement with outcomes $y\in\mathcal Y$, write

```math
p_\theta(y\mid x).
```

The current Effective Dimension implementation uses computational-basis
measurement outcomes.  In that implementation the generic output variable is
therefore identified with a bitstring $z$:

```math
y=z,
\qquad
p_\theta(z\mid x)
=
|\langle z|\psi_\theta(x)\rangle|^2.
```

This choice is part of the statistical model and must be kept explicit when
CFIM-based quantities are compared with state-space quantities.

## Model output and Jacobian

For QNTK let the selected scalar model output be

```math
f_\theta(x).
```

For samples $x_a$ define the output Jacobian

```math
J_{a i}
=
\frac{\partial f_\theta(x_a)}{\partial\theta_i}.
```

The output function is part of the definition of the kernel.  Two observables
on the same quantum state may therefore induce different QNTKs.

## Parameter-space distributions for Effective Dimension

Write $\pi(\theta)$ for a distribution over parameters.

The current repository uses:

- GED: independent uniform sampling over the selected global parameter domain,
  currently $[0,2\pi)$ for every scalar parameter;
- LED: uniform sampling over a Euclidean $\epsilon$-ball centred at the current
  parameter vector $\theta_0$.

The theoretical dataset-size variable in GED/LED is denoted $n$; it is not the
same object as the number $N$ of inputs used to estimate an empirical Fisher
matrix.
