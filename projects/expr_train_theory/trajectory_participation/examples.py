import numpy as np


def ket(index, dimension):
    state = np.zeros(dimension, dtype=complex)
    state[index] = 1.0
    return state


def identical_states(count=4, dimension=4):
    state = ket(0, dimension)
    return np.stack([state] * count)


def orthogonal_states(count=4):
    return np.eye(count, dtype=complex)


def duplicated_orthogonal_states():
    # multiplicities 2, 1, 1 -> d_TP = 16 / 6 = 8/3
    return np.stack([ket(0, 3), ket(0, 3), ket(1, 3), ket(2, 3)])


def nearly_collinear_states(epsilon=1e-2):
    e0 = ket(0, 2)
    e1 = ket(1, 2)
    second = e0 + epsilon * e1
    second /= np.linalg.norm(second)
    return np.stack([e0, second])
