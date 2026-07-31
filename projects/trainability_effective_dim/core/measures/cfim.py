import torch

def cfim_for_input(net, inputs, min_probability=1e-12):
    weights = net.qlayers.weights
    flat_weights = weights.reshape(-1)
    n_params = flat_weights.numel()

    state = net.state_circuit(inputs, weights)
    probabilities = torch.abs(state) ** 2
    probabilities = probabilities.reshape(-1)

    fisher = torch.zeros(
        (n_params, n_params),
        dtype=weights.dtype,
        device=weights.device,
    )

    for probability in probabilities:
        if probability.detach() > min_probability:
            gradient = torch.autograd.grad(
                probability,
                weights,
                retain_graph=True,
            )[0].reshape(-1)

            fisher += torch.outer(gradient, gradient) / probability

    return fisher.detach(), probabilities.detach()