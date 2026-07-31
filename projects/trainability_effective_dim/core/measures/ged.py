import torch
import math

from .monte_carlo import sample_empirical_fishers
from .samplers import UniformParameterSpace

def estimate_global_effective_dimension(
            net,
            inputs,
            n_theta=100,
            min_probability=1e-12,
            array_of_theoretical_number_of_data_samples=None,
    ):
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