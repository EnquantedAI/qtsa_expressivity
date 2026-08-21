from .core import trajectory_participation_dimension
from .examples import (
    duplicated_orthogonal_states,
    identical_states,
    nearly_collinear_states,
    orthogonal_states,
)


def show(name, states):
    result = trajectory_participation_dimension(states)
    print(f"{name:24s} d_TP={result.dimension:.8f} rank={result.numerical_rank}")
    print("  spectrum:", result.spectrum)


def main():
    show("identical states", identical_states())
    show("orthogonal states", orthogonal_states())
    show("duplicate + orthogonal", duplicated_orthogonal_states())
    show("nearly collinear", nearly_collinear_states())


if __name__ == "__main__":
    main()
