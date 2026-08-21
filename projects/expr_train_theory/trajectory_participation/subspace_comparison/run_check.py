import numpy as np

from .basis import compare_hard_and_soft_dimension


def _show(name, states):
    result = compare_hard_and_soft_dimension(states)
    print(name)
    print(f"  Gram-Schmidt rank: {result.gram_schmidt_rank}")
    print(f"  SVD rank:          {result.svd_rank}")
    print(f"  dTP:               {result.participation_dimension:.12f}")
    print(f"  dTP / rank:        {result.rank_utilization:.12f}")


def main():
    e0 = np.array([1.0, 0.0])
    e1 = np.array([0.0, 1.0])

    _show("balanced two-dimensional trajectory", [e0, e1])
    _show("same subspace, biased sampling", [e0, e0, e0, e1])


if __name__ == "__main__":
    main()
