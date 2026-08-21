import numpy as np

from ..sampling import qubit_arc_states, weighted_trajectory_participation_dimension
from .weights import arc_length_snapshot_weights, arc_length_weighted_participation_dimension


def main():
    sample_sets = {
        "coarse_uniform": np.linspace(0.0, np.pi / 2.0, 5),
        "dense_uniform": np.linspace(0.0, np.pi / 2.0, 21),
        "nonuniform": np.array([0.0, 0.04, 0.12, 0.35, 0.9, 1.2, np.pi / 2.0]),
    }

    for name, parameters in sample_sets.items():
        states = qubit_arc_states(parameters)
        equal = weighted_trajectory_participation_dimension(states)
        weights = arc_length_snapshot_weights(states)
        weighted = arc_length_weighted_participation_dimension(states)
        print(name)
        print(f"  snapshots:           {len(parameters)}")
        print(f"  equal-weight dTP:    {equal:.12f}")
        print(f"  arc-length dTP:      {weighted:.12f}")
        print(f"  min/max arc weight:  {weights.min():.6f} / {weights.max():.6f}")


if __name__ == "__main__":
    main()
