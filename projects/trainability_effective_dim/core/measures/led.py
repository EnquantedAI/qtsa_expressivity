import math
import torch

from .monte_carlo import sample_empirical_fishers
from .samplers import EpsilonBallParameterSpace


def estimate_local_effective_dimension(
    net,
    inputs,
    epsilon=0.1,
    n_theta=100,
    min_probability=1e-12,
    array_of_theoretical_number_of_data_samples=None,
):
    """
    Estimate the local effective dimension (LED) of a quantum model.

    LED measures the number of locally distinguishable parameter directions
    near the model's current parameter values. It samples parameter vectors
    uniformly from an epsilon-ball centred at `net.qlayers.weights`.

    For every sampled parameter vector theta_k, the function estimates an
    empirical Fisher matrix F(theta_k) by averaging the CFIM over `inputs`.
    It normalizes the Fisher matrices as:

        F_hat(theta_k) =
            d * F(theta_k) / Tr(mean_k(F(theta_k))),

    where d is the total number of trainable parameters.

    For each theoretical dataset size n, LED is estimated with:

        kappa = n / (2 * pi * log(n))

        LED(n) =
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
            - `net.qlayers.weights`, the current quantum-layer weights;
            - the interface required by `sample_empirical_fishers`.

        inputs:
            Input samples used to average each sampled parameter vector's
            CFIM. The expected shape is typically
            `(number_of_inputs, input_dimension)`.

        epsilon:
            Radius of the local Euclidean parameter neighbourhood around the
            current model weights. Parameter vectors are sampled uniformly
            over this epsilon-ball. Default: 0.1.

        n_theta:
            Number of parameter vectors sampled from the epsilon-ball.
            Default: 100.

        min_probability:
            Probability threshold passed to the CFIM estimator. Outcomes at
            or below this threshold are ignored to avoid numerical instability.
            Default: 1e-12.

        array_of_theoretical_number_of_data_samples:
            Iterable of theoretical dataset sizes n for which LED is computed.
            If omitted, LED is evaluated only for `len(inputs)`.

    Returns:
        list[float]:
            One local effective-dimension estimate for each requested
            theoretical dataset size, in the same order as
            `array_of_theoretical_number_of_data_samples`.

    Raises:
        ValueError:
            If a theoretical dataset size is at most one.

        ValueError:
            If the trace of the mean empirical Fisher matrix is not positive.

        ValueError:
            If kappa is at most one. This prevents invalid or negative-looking
            values caused by division by log(kappa) when log(kappa) <= 0.

    Notes:
        - The epsilon-ball is centred at the model weights at function entry.
        - `sample_empirical_fishers` returns one input-averaged empirical
          Fisher matrix for each sampled parameter vector.
        - The parameter-space Monte Carlo average is performed after the
          nonlinear log-determinant calculation.
        - For meaningful positive values, use theoretical dataset sizes for
          which kappa > 1.
    """
    if array_of_theoretical_number_of_data_samples is None:
        array_of_theoretical_number_of_data_samples = [len(inputs)]

    d = net.parameter_count

    parameter_space = EpsilonBallParameterSpace(
        center=net.qlayers.weights,
        epsilon=epsilon,
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

        led = (
            2.0
            * (torch.logsumexp(log_integrand, dim=0) - math.log(n_theta))
            / math.log(kappa)
        )

        effective_dimensions.append(led.item())

    return effective_dimensions