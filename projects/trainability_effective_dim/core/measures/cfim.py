import torch


def cfim_for_input(net, inputs, min_probability=1e-12):
    """
    Compute the classical Fisher information matrix for one model input.

    The function evaluates the quantum state produced by `net` for fixed
    `inputs` and the model's current trainable quantum-layer weights. It
    interprets squared state-vector amplitudes as computational-basis outcome
    probabilities and computes

        F_ij = sum_z [
            (d p_z / d theta_i) * (d p_z / d theta_j) / p_z
        ],

    where z ranges over computational-basis measurement outcomes, theta_i and
    theta_j are flattened model parameters, and p_z is the probability of
    outcome z.

    Outcomes with probability at most `min_probability` are excluded to avoid
    division by zero and numerical instability.

    Args:
        net:
            Model instance with:
            - `net.qlayers.weights`: trainable PyTorch tensor containing the
              quantum-circuit parameters.
            - `net.state_circuit(inputs, weights)`: method returning the
              state vector for the supplied input and parameter tensor.

        inputs:
            One input sample accepted by `net.state_circuit`.

        min_probability:
            Outcomes with probabilities less than or equal to this threshold
            are ignored. Default: 1e-12.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            fisher:
                Detached classical Fisher information matrix with shape
                `(number_of_parameters, number_of_parameters)`.

            probabilities:
                Detached flattened vector of computational-basis
                probabilities. For `n` qubits, it has length `2 ** n`.

    Notes:
        - The matrix is computed with respect to every scalar entry in
          `net.qlayers.weights`.
        - PyTorch autograd computes each probability gradient.
        - The returned tensors are detached and cannot be differentiated
          through further.
        - This implementation assumes that `state_circuit` returns a state
          vector. If it returns measurement probabilities directly, remove
          the `torch.abs(state) ** 2` operation.
    """
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