from pennylane import numpy as np
import torch
from .cfim import cfim_for_input

def sample_empirical_fishers(
    net,
    parameter_space,
    inputs,
    n_theta=100,
    min_probability=1e-12,
):
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