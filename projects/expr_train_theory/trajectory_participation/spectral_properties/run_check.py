import numpy as np

from .metrics import spectral_dimension_profile


def main():
    states = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=complex,
    )
    profile = spectral_dimension_profile(states)
    print(f"numerical rank:          {profile.numerical_rank}")
    print(f"stable rank:             {profile.stable_rank:.12f}")
    print(f"dTP = exp(H_2):          {profile.participation_dimension:.12f}")
    print(f"entropy dimension H_1:   {profile.entropy_dimension:.12f}")
    print(
        "ordering:                "
        f"{profile.stable_rank:.6f} <= {profile.participation_dimension:.6f} "
        f"<= {profile.entropy_dimension:.6f} <= {profile.numerical_rank}"
    )


if __name__ == "__main__":
    main()
