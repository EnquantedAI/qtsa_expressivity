import torch

from .cfim import cfim_for_input


def sample_empirical_fishers(
    net,
    parameter_space,
    inputs,
    n_theta=100,
    min_probability=1e-12,
):
    """
    Estimate empirical classical Fisher information matrices by Monte Carlo
    sampling over a parameter-space distribution.

    For each sampled parameter vector theta_k, the function evaluates the CFIM
    for every input x_j and returns the input-averaged empirical Fisher matrix:

        F_emp(theta_k) = (1 / N_x) * sum_j F(theta_k, x_j).

    The procedure is repeated `n_theta` times. The returned tensor therefore
    retains one empirical Fisher matrix per sampled parameter vector, which is
    required by global and local effective-dimension estimators.

    Args:
        net:
            Quantum model instance with:
            - `net.qlayers.weights`, a trainable PyTorch tensor;
            - `net.state_circuit(inputs, weights)`, used by `cfim_for_input`.

        parameter_space:
            Object that implements `sample()` and returns one parameter tensor.
            Each sample must be reshapeable to `net.qlayers.weights.shape`.
            Examples include `UniformParameterSpace` and
            `EpsilonBallParameterSpace`.

        inputs:
            Input tensor with shape `(N_x, input_dimension)`, where N_x is the
            number of input samples. A one-dimensional tensor is interpreted
            as one input sample and receives a batch dimension automatically.

        n_theta:
            Number of parameter vectors drawn from `parameter_space`.
            Default: 100.

        min_probability:
            Probability threshold forwarded to `cfim_for_input`. Outcomes at
            or below this threshold are ignored in the CFIM calculation.
            Default: 1e-12.

    Returns:
        torch.Tensor:
            Empirical Fisher matrices with shape:

                (n_theta, number_of_parameters, number_of_parameters)

            Entry `result[k]` is the CFIM averaged over all supplied inputs
            for the k-th sampled parameter vector.

    Notes:
        - The original weights are copied before sampling and restored in the
          `finally` block, including if CFIM calculation raises an exception.
        - `torch.no_grad()` prevents sampled-weight assignment and restoration
          from being tracked by autograd. [web:180]
        - `torch.stack` adds the leading sample dimension when assembling
          identically shaped Fisher matrices. [web:183]
        - This function does not average over `n_theta`; GED and LED must
          perform their own parameter-space Monte Carlo average after applying
          their nonlinear log-determinant integrand.
    """
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)

    original_weights = net.qlayers.weights.detach().clone()
    fisher_matrices = []

    try:
        for _ in range(n_theta):
            sampled_weights = parameter_space.sample()

            sampled_weights = torch.as_tensor(
                sampled_weights,
                dtype=original_weights.dtype,
                device=original_weights.device,
            ).reshape_as(original_weights)

            with torch.no_grad():
                net.qlayers.weights.copy_(sampled_weights)

            cfims_for_theta = []

            for x in inputs:
                cfim, _ = cfim_for_input(
                    net,
                    x,
                    min_probability=min_probability,
                )
                cfims_for_theta.append(cfim)

            empirical_fisher = torch.stack(cfims_for_theta).mean(dim=0)
            fisher_matrices.append(empirical_fisher)

    finally:
        with torch.no_grad():
            net.qlayers.weights.copy_(original_weights)

    return torch.stack(fisher_matrices)