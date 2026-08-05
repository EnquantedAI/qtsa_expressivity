# Effective Dimension Workflow

This implementation estimates the **global effective dimension (GED)** and
**local effective dimension (LED)** of a parameterized quantum model.

```text
Quantum model
    |
    v
CFIM for one input and one parameter vector
    |
    v
Monte Carlo averaging over inputs and sampled parameters
    |
    +--> GED: global uniform parameter sampling
    |
    +--> LED: local epsilon-ball parameter sampling
```

## 1. Classical Fisher Information Matrix

`cfim_for_input(net, inputs)` computes the **classical Fisher information
matrix (CFIM)** for one fixed parameter tensor $\theta$ and one fixed model
input $x$.

The quantum circuit returns a state vector:

$$
|\psi_\theta(x)\rangle.
$$

The probability of each computational-basis measurement outcome $z$ is:

$$
p_\theta(z \mid x)
=
\left|
\langle z \mid \psi_\theta(x)\rangle
\right|^2.
$$

For each non-negligible outcome probability, PyTorch autograd calculates the
gradient with respect to all circuit parameters:

$$
\nabla_\theta p_\theta(z \mid x).
$$

The CFIM is:

$$
F(\theta, x)
=
\sum_{z}
\frac{
\nabla_\theta p_\theta(z \mid x)
\nabla_\theta p_\theta(z \mid x)^\top
}{
p_\theta(z \mid x)
}.
$$

The output has shape:

```text
(number_of_parameters, number_of_parameters)
```

The `min_probability` threshold excludes outcomes with nearly zero probability
to avoid division by zero and numerical instability.

## 2. Parameter samplers

The Monte Carlo estimator needs a parameter sampler. Both samplers implement:

```python
sampled_weights = parameter_space.sample()
```

The returned tensor must have the same shape as:

```python
net.qlayers.weights
```

### UniformParameterSpace

`UniformParameterSpace` samples every parameter independently:

$$
\theta_i \sim \mathrm{Uniform}(\mathrm{low}, \mathrm{high}).
$$

For GED, the usual global domain is:

$$
\theta_i \sim \mathrm{Uniform}(0, 2\pi).
$$

This samples throughout a hypercube in parameter space.

### EpsilonBallParameterSpace

`EpsilonBallParameterSpace` samples near a centre $\theta_0$, usually the
model's current weights:

$$
\|\theta - \theta_0\|_2 \leq \epsilon.
$$

It samples a random unit direction $u$ and a radius:

$$
r = \epsilon\, q^{1/d},
\qquad
q \sim \mathrm{Uniform}(0, 1),
$$

where $d$ is the total number of parameters. It then constructs:

$$
\theta = \theta_0 + r u.
$$

The exponent $1/d$ ensures samples are uniform over the **volume** of the
epsilon-ball rather than being concentrated near its centre.

## 3. Monte Carlo Fisher matrices

`sample_empirical_fishers(...)` combines a parameter sampler with CFIM
calculation.

For each sampled parameter vector $\theta_k$:

1. The sampled weights replace `net.qlayers.weights`.
2. `cfim_for_input` is evaluated for every input $x_j$.
3. The CFIMs are averaged over inputs:

$$
F_{\mathrm{emp}}(\theta_k)
=
\frac{1}{N_x}
\sum_{j=1}^{N_x}
F(\theta_k, x_j).
$$

4. The empirical Fisher matrix is stored.
5. The original model weights are restored after sampling.

The returned tensor has shape:

```text
(n_theta, number_of_parameters, number_of_parameters)
```

Thus, `fishers[k]` is the input-averaged empirical Fisher matrix at the
$k$-th sampled parameter vector.

The function intentionally does **not** average over `n_theta`, because GED
and LED apply a nonlinear log-determinant expression to every Fisher matrix
before the final parameter-space average.

## 4. Fisher normalization

Both LED and GED normalize the empirical Fisher matrices as:

$$
\hat F(\theta_k)
=
\frac{
d F_{\mathrm{emp}}(\theta_k)
}{
\mathrm{Tr}
\left[
\frac{1}{N_\theta}
\sum_{k=1}^{N_\theta}
F_{\mathrm{emp}}(\theta_k)
\right]
}.
$$

Here, $d$ is the number of trainable parameters. This makes the estimate
invariant under multiplying every Fisher matrix by the same positive constant.

## 5. Effective dimension

For a theoretical dataset size $n$, define:

$$
\kappa
=
\frac{n}{2\pi \log(n)}.
$$

For each sampled parameter vector, compute:

$$
r_k
=
\frac{1}{2}
\log
\det
\left[
I + \kappa \hat F(\theta_k)
\right].
$$

The effective dimension is estimated by Monte Carlo integration:

$$
d_{\mathrm{eff}}(n)
=
\frac{2}{\log(\kappa)}
\log
\left[
\frac{1}{N_\theta}
\sum_{k=1}^{N_\theta}
\exp(r_k)
\right].
$$

The implementation uses `torch.linalg.slogdet` and `torch.logsumexp` for
numerical stability:

```python
log_determinants = torch.linalg.slogdet(
    identity.unsqueeze(0) + kappa * f_hat
).logabsdet

log_integrand = 0.5 * log_determinants

effective_dimension = (
    2.0
    * (torch.logsumexp(log_integrand, dim=0) - math.log(n_theta))
    / math.log(kappa)
)
```

Use theoretical dataset sizes for which $\kappa > 1$.

## 6. Global effective dimension

`estimate_global_effective_dimension(...)` uses:

```python
parameter_space = UniformParameterSpace(
    low=0.0,
    high=2.0 * math.pi,
    shape=net.qlayers.weights.shape,
    dtype=net.qlayers.weights.dtype,
)
```

This estimates effective dimension over the full parameter domain:

$$
\theta \sim \mathrm{Uniform}([0, 2\pi)^d).
$$

GED answers:

> Across the whole chosen parameter space, how many parameter directions are distinguishable from the model output distribution?

## 7. Local effective dimension

`estimate_local_effective_dimension(...)` uses:

```python
parameter_space = EpsilonBallParameterSpace(
    center=net.qlayers.weights,
    epsilon=epsilon,
)
```

This estimates effective dimension near the current model parameters:

$$
\theta \sim
\mathrm{Uniform}
\left(
B_\epsilon(\theta_0)
\right).
$$

LED answers:

> Near the current model weights, how many parameter directions are locally distinguishable from the model output distribution?

The epsilon parameter controls locality:

- Small `epsilon`: highly local analysis near current weights.
- Large `epsilon`: wider local neighbourhood.
- Uniform global sampling: analysis across the full parameter domain.

## 8. End-to-end usage

```python
inputs = torch.randn(
    100,
    input_dimension,
    dtype=net.qlayers.weights.dtype,
)
```

Global effective dimension:

```python
ged = estimate_global_effective_dimension(
    net=net,
    inputs=inputs,
    n_theta=100,
    array_of_theoretical_number_of_data_samples=[100, 1_000, 10_000],
)
```

Local effective dimension:

```python
led = estimate_local_effective_dimension(
    net=net,
    inputs=inputs,
    epsilon=0.1,
    n_theta=100,
    array_of_theoretical_number_of_data_samples=[100, 1_000, 10_000],
)
```

Both functions return one effective-dimension value for each requested
theoretical dataset size.