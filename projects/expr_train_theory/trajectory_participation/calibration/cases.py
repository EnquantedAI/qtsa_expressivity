import numpy as np


def basis_state(index, dimension):
    state = np.zeros(dimension, dtype=complex)
    state[index] = 1.0
    return state


def identical_states(count=4, dimension=4):
    state = basis_state(0, dimension)
    return np.asarray([state.copy() for _ in range(count)])


def orthogonal_states(count=4, dimension=4):
    if count > dimension:
        raise ValueError("count cannot exceed dimension")
    return np.asarray([basis_state(i, dimension) for i in range(count)])


def duplicated_direction_case():
    e0 = basis_state(0, 3)
    e1 = basis_state(1, 3)
    return np.asarray([e0, e0, e1])


def nearly_collinear_case(epsilon=1e-2):
    e0 = basis_state(0, 3)
    e1 = basis_state(1, 3)
    tilted = e0 + epsilon * e1
    tilted /= np.linalg.norm(tilted)
    return np.asarray([e0, tilted, e1])
