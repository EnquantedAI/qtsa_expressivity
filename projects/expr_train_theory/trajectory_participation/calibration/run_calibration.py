import csv
from pathlib import Path

from .cases import (
    duplicated_direction_case,
    identical_states,
    nearly_collinear_case,
    orthogonal_states,
)
from .metrics import calibrate_trajectory


def main():
    cases = {
        "identical": identical_states(),
        "orthogonal": orthogonal_states(),
        "duplicated_direction": duplicated_direction_case(),
        "nearly_collinear": nearly_collinear_case(),
    }

    rows = []
    for name, states in cases.items():
        result = calibrate_trajectory(states)
        row = {
            "case": name,
            "d_tp": result.dimension,
            "rank": result.numerical_rank,
            "ceiling": result.ceiling,
            "fraction_of_rank": result.fraction_of_rank,
            "fraction_of_ceiling": result.fraction_of_ceiling,
            "entropy_dimension": result.entropy_dimension,
            "stable_rank": result.stable_rank,
            "largest_weight": result.largest_weight,
        }
        rows.append(row)
        print(
            f"{name:20s} dTP={result.dimension:.6f}  "
            f"rank={result.numerical_rank}/{result.ceiling}  "
            f"dTP/rank={result.fraction_of_rank:.6f}"
        )

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "trajectory_calibration.csv"
    with out_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
