import numpy as np

from ..core import trajectory_participation_dimension
from .metrics import (
    first_state_frame_potential,
    trajectory_density_purity_from_overlaps,
    trajectory_participation_dimension_from_overlaps,
)


def main():
    states = np.array(
        [
            [1.0, 0.0],
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            [0.0, 1.0],
        ],
        dtype=complex,
    )

    direct = trajectory_participation_dimension(states).dimension
    overlap = trajectory_participation_dimension_from_overlaps(states)
    fp1 = first_state_frame_potential(states)
    purity = trajectory_density_purity_from_overlaps(states)

    print(f"state frame potential: {fp1:.12f}")
    print(f"trajectory purity:     {purity:.12f}")
    print(f"dTP from SVD:          {direct:.12f}")
    print(f"dTP from overlaps:     {overlap:.12f}")


if __name__ == "__main__":
    main()
