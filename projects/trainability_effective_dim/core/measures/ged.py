import math
import torch

from .monte_carlo import sample_empirical_fishers
from .samplers import UniformParameterSpace


def estimate_global_effective_dimension(
    net,
    inputs,
    n_theta=100,
    min_probability=1e-12,
    array_of_theoretical_number_of_data_samples=None,
):
    """
    Estimate the global effective dimension (GED) of a quantum model.

    GED estimates the number of distinguishable parameter directions across the
    entire chosen parameter domain. This implementation samples parameter
    tensors uniformly from [0, 2*pi) for every scalar circuit parameter.

    For every sampled parameter vector theta_k, the function first estimates
    an input-averaged empirical classical Fisher information matrix F(theta_k).
    The CFIM quantifies how circuit-output probabilities change with the model
    parameters.

    The Fisher matrices are normalized as:

        F_hat(theta_k) =
            d * F(theta_k) / Tr(mean_k(F(theta_k))),

    where d is the total number of trainable parameters.

    For each theoretical dataset size n, GED is estimated by:

        kappa = n / (2 * pi * log(n))

        GED(n) =
            2 / log(kappa) *
            log(
                (1 / N_theta) *
                sum_k sqrt(det(I + kappa * F_hat(theta_k)))
            ).

    Args:
        net:
            Quantum model instance with:
            - `net.parameter_count`, the number of scalar trainable
              parameters;
            - `net.qlayers.weights`, the quantum-layer weight tensor;
            - the interface required by `sample_empirical_fishers`.

        inputs:
            Input samples used to average the CFIM for each sampled parameter
            tensor. The expected shape is typically
            `(number_of_inputs, input_dimension)`.

        n_theta:
            Number of parameter tensors sampled from the global uniform
            parameter space. Default: 100.

        min_probability:
            Probability threshold passed to the CFIM estimator. Outcomes at
            or below this threshold are ignored. Default: 1e-12.

        array_of_theoretical_number_of_data_samples:
            Iterable of theoretical dataset sizes n for which GED is computed.
            If omitted, GED is evaluated only for `len(inputs)`.

    Returns:
        list[float]:
            One GED estimate per requested theoretical dataset size, in the
            same order as `array_of_theoretical_number_of_data_samples`.

    Raises:
        ValueError:
            If a theoretical dataset size is at most one.

        ValueError:
            If the trace of the mean empirical Fisher matrix is not positive.

        ValueError:
            If kappa is at most one. This prevents invalid or negative-looking
            values caused by division by log(kappa) when log(kappa) <= 0.

    Notes:
        - The global parameter domain is the hypercube [0, 2*pi)^d.
        - For another global measure, pass a different sampler or expose the
          parameter space as a function argument.
        - `sample_empirical_fishers` must return shape `(n_theta, d, d)`;
          its leading dimension is retained until after the log-determinant
          calculation.
    """
    if array_of_theoretical_number_of_data_samples is None:
        array_of_theoretical_number_of_data_samples = [len(inputs)]

    d = net.parameter_count
    weights = net.qlayers.weights

    parameter_space = UniformParameterSpace(
        low=0.0,
        high=2.0 * math.pi,
        shape=weights.shape,
        dtype=weights.dtype,
    )

    fishers = sample_empirical_fishers(
        net=net,
        parameter_space=parameter_space,
        inputs=inputs,
        n_theta=n_theta,
        min_probability=min_probability,
    )

    mean_fisher = fishers.mean(dim=0)
    mean_fisher_trace = torch.trace(mean_fisher)

    if mean_fisher_trace <= 0:
        raise ValueError("Mean Fisher trace must be positive.")

    f_hat = d * fishers / mean_fisher_trace

    identity = torch.eye(
        d,
        dtype=fishers.dtype,
        device=fishers.device,
    )

    effective_dimensions = []

    for n in array_of_theoretical_number_of_data_samples:
        if n <= 1:
            raise ValueError("Theoretical dataset size n must exceed 1.")

        kappa = n / (2.0 * math.pi * math.log(n))

        if kappa <= 1.0:
            raise ValueError(
                "Theoretical dataset size is too small: kappa must exceed 1."
            )

        log_determinants = torch.linalg.slogdet(
            identity.unsqueeze(0) + kappa * f_hat
        ).logabsdet

        log_integrand = 0.5 * log_determinants

        ged = (
            2.0
            * (torch.logsumexp(log_integrand, dim=0) - math.log(n_theta))
            / math.log(kappa)
        )

        effective_dimensions.append(ged.item())

    return effective_dimensions