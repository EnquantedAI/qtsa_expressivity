# trainability_effective_dim

QNN-based time-series forecasting with expressivity and trainability analysis
via global effective dimension (GED), and local effective dimension (LED).

## How to run

Open `experiment_setup.ipynb` and execute the cells sequentially.

You may skip the Optuna hyperparameter-search cells because they are
computationally expensive. Effective hyperparameters are already provided to
`BasicTrainer`.

The main pipeline is:

```text
data generation
    → input-output pair creation
    → optional Optuna hyperparameter search
    → model training
    → simple and continuous forecasting
    → loss visualization
    → GED / LED analysis
```

## Structure

```text
trainability_effective_dim/
├── core/
│   ├── model/
│   │   ├── docs.md                   # Documentation for GQNN model
│   │   └── models.py                 # GQNN quantum neural network model
│   ├── measures/
│   │   ├── docs.md                   # Documentation for the measures model
│   │   ├── cfim.py                   # Classical Fisher information matrix
│   │   ├── samplers.py               # Uniform and epsilon-ball samplers
│   │   ├── monte_carlo.py            # Empirical Fisher Monte Carlo estimator
│   │   ├── ged.py                    # Global effective dimension
│   │   └── led.py                    # Local effective dimension
│   ├── trainers/
│   │   ├── abstract_trainer.py       # AbstractTrainer base class
│   │   ├── basic_trainer.py          # Training and forecasting
│   │   ├── hyperparameter_trainer.py # [WIP] Optuna hyperparameter search
│   │   └── statistical_trainer.py    # [WIP] Multiple-run statistics
│   └── datasets/                     # Preprocessed .pt datasets
├── references/                       # Background papers on effective dimension
├── results/                          # Figures and hyperparameter-search results
├── tests/                            # CFIM, sampler, Monte Carlo, GED, LED tests
├── experiment_setup.ipynb            # End-to-end experiment pipeline
└── README.md
```

## Dependencies

This project depends on utilities elsewhere in `qtsa_expressivity`:

- `src.datagen` — Mackey-Glass and NARMA10 time-series generation;
- `src.models` — circuit drawing, `gqnn_shape`, and `zz_feature_map`;
- `projects.expr_train_theory.qfim` — pure-state QFIM calculations and
  diagnostics;
- `utils.Charts` — plotting helpers.

## Pipeline details

1. **Data preparation** — Mackey-Glass and NARMA10 series are generated with
   `src.datagen`, split chronologically into train/validation/test chunks,
   scaled using training-only statistics, converted into sliding-window
   input-output pairs, and saved as PyTorch `TensorDataset`s.

2. **Model** — `GQNN` in `core/model/models.py` defines a variational quantum
   circuit. Classical inputs are embedded with a configurable feature map
   such as ZZ, IQP, or `AngleEmbedding`, then processed by
   `StronglyEntanglingLayers`.

3. **Training** — `BasicTrainer` runs Adam-based training with optional L1/L2
   regularization. `HyperparameterTrainer` is intended to run Optuna trials
   with pruning but remains work in progress.

4. **Forecasting** — `BasicTrainer` supports:
   - `test_simple_forecast`: one-step prediction using ground-truth windows;
   - `test_continuous_forecast`: autoregressive forecasting, where each
     prediction becomes part of the next input window.

5. **Effective-dimension analysis** — CFIM, Monte Carlo estimation, GED, and
   LED can be applied to a model and a representative set of inputs after
   model initialization or training.

## Further documentation

Detailed docstrings are available directly in:

- `core/model/models.py` for the model architecture and circuit interfaces;
- `core/measures/cfim.py` for CFIM computation;
- `core/measures/samplers.py` for global and local parameter sampling;
- `core/measures/monte_carlo.py` for empirical Fisher estimation;
- `core/measures/ged.py` for global effective dimension;
- `core/measures/led.py` for local effective dimension;
- `tests/` for documented expected behavior and analytic reference cases.

## Status

- CFIM, empirical Fisher estimation, parameter samplers, GED, LED, and their
  unit tests are implemented.
- `core/trainers/statistical_trainer.py` is not yet implemented.
- `core/trainers/hyperparameter_trainer.py` is not fully implemented.
- `experiment_setup.ipynb` remains work in progress.


## Effective dimension workflow

The effective-dimension implementation consists of four layers:

```text
Quantum model statevector
    ↓
CFIM for one input and one parameter vector
    ↓
Monte Carlo sampling over inputs and parameters
    ↓
GED: global uniform parameter sampling
LED: local epsilon-ball parameter sampling
```

### Classical Fisher information matrix

`core/measures/cfim.py` provides:

```python
cfim_for_input(net, inputs)
```

It computes the CFIM for one fixed input $x$ and the model's current quantum
parameters $\theta$.

For every computational-basis outcome $z$, the statevector is converted into a
probability:

$$
p_\theta(z \mid x)
=
\left|
\langle z \mid \psi_\theta(x)\rangle
\right|^2.
$$

The CFIM is:

$$
F(\theta, x)
=
\sum_z
\frac{
\nabla_\theta p_\theta(z \mid x)
\nabla_\theta p_\theta(z \mid x)^\top
}{
p_\theta(z \mid x)
}.
$$

Outcomes with probability below `min_probability` are ignored to prevent
division by zero and numerical instability.

### Parameter samplers

`core/measures/samplers.py` defines parameter-space samplers. Each sampler
implements:

```python
sampled_weights = parameter_space.sample()
```

The returned tensor must have the same shape as `net.qlayers.weights`.

#### Uniform parameter space

`UniformParameterSpace` independently samples every parameter from a
half-open interval:

$$
\theta_i \sim \mathrm{Uniform}(\mathrm{low}, \mathrm{high}).
$$

GED uses the global parameter domain:

$$
\theta_i \sim \mathrm{Uniform}(0, 2\pi).
$$

#### Epsilon-ball parameter space

`EpsilonBallParameterSpace` samples uniformly from a local Euclidean ball
around a centre $\theta_0$:

$$
\|\theta - \theta_0\|_2 \leq \epsilon.
$$

A sample is generated from a random unit direction $u$ and radius:

$$
r = \epsilon q^{1/d},
\qquad
q \sim \mathrm{Uniform}(0, 1),
$$

where $d$ is the total number of scalar parameters. The result is:

$$
\theta = \theta_0 + r u.
$$

The $1/d$ exponent ensures uniform sampling over the volume of the ball.

### Monte Carlo empirical Fisher estimation

`core/measures/monte_carlo.py` provides:

```python
sample_empirical_fishers(
    net,
    parameter_space,
    inputs,
    n_theta=100,
)
```

For every sampled parameter vector $\theta_k$, the function computes a CFIM
for each input $x_j$ and averages over inputs:

$$
F_{\mathrm{emp}}(\theta_k)
=
\frac{1}{N_x}
\sum_{j=1}^{N_x}
F(\theta_k, x_j).
$$

The result has shape:

```text
(n_theta, number_of_parameters, number_of_parameters)
```

The original model weights are restored after sampling, including when an
exception occurs.

### Fisher normalization

Both LED and GED normalize empirical Fisher matrices as:

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
},
$$

where $d$ is the number of trainable parameters.

### Global effective dimension

`core/measures/ged.py` provides:

```python
estimate_global_effective_dimension(
    net,
    inputs,
    n_theta=100,
    array_of_theoretical_number_of_data_samples=None,
)
```

GED samples uniformly across the entire selected parameter domain. It answers:

> Across the complete parameter space, how many parameter directions are
> distinguishable through the model output distribution?

### Local effective dimension

`core/measures/led.py` provides:

```python
estimate_local_effective_dimension(
    net,
    inputs,
    epsilon=0.1,
    n_theta=100,
    array_of_theoretical_number_of_data_samples=None,
)
```

LED samples from an epsilon-ball centred at the model's current weights. It
answers:

> Near the current trained parameter vector, how many parameter directions are
> locally distinguishable through the model output distribution?

For both GED and LED, for theoretical dataset size $n$:

$$
\kappa = \frac{n}{2\pi \log(n)}.
$$

The effective dimension is estimated as:

$$
d_{\mathrm{eff}}(n)
=
\frac{2}{\log(\kappa)}
\log
\left[
\frac{1}{N_\theta}
\sum_{k=1}^{N_\theta}
\sqrt{
\det
\left[
I + \kappa \hat F(\theta_k)
\right]
}
\right].
$$

Use values of $n$ for which $\kappa > 1$, typically beginning from
`n = 100`.


## Tests

The `tests/` directory contains unit tests for all implemented
effective-dimension components.

```text
tests/
├── test_cfim.py
├── test_samplers.py
├── test_monte_carlo.py
├── test_ged.py
└── test_led.py
```

### CFIM tests

`test_cfim.py` verifies known analytic circuits:

- a single `RY` rotation has CFIM $[[1]]$;
- two parameters affecting the same rotation produce a rank-one CFIM;
- independent single-qubit rotations produce an identity CFIM;
- exactly cancelling rotations produce a zero CFIM.

### Sampler tests

`test_samplers.py` verifies:

- expected shape and dtype;
- uniform-sampler parameter bounds;
- epsilon-ball distance constraints;
- immutable copied epsilon-ball centre;
- reproducibility with seeded random generators;
- rejection of non-positive epsilon;
- uniform transformed radii;
- isotropic sampled directions.

### Monte Carlo tests

`test_monte_carlo.py` verifies:

- empirical Fisher averaging across several inputs;
- operation with both uniform and epsilon-ball parameter samplers;
- restoration of original model weights;
- support for one-dimensional single-input tensors.

### GED and LED tests

`test_ged.py` and `test_led.py` verify:

- known one-dimensional analytic effective-dimension values;
- correct per-parameter-sample log-determinant evaluation;
- invariance under global positive rescaling of Fisher matrices;
- one output per requested theoretical dataset size;
- validation of invalid dataset sizes;
- correct use of global uniform sampling for GED;
- correct use of epsilon-ball sampling for LED.

Run all tests from the repository root:

```bash
python -m unittest discover -v
```

Alternatively, run an individual test module:

```bash
python -m unittest projects.trainability_effective_dim.tests.test_cfim -v
```