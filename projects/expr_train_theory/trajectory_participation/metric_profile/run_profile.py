from __future__ import annotations

import argparse
from pathlib import Path

from .profile import default_profiles, save_profiles


def main():
    parser = argparse.ArgumentParser(description="Compare dTP, QFIM, CFIM and QNTK on small reference cases.")
    parser.add_argument("--save", action="store_true", help="save CSV/JSON results")
    args = parser.parse_args()

    profiles = default_profiles()
    header = (
        "case",
        "dTP",
        "traj-rank",
        "QFIM-rank",
        "CFIM-rank",
        "QNTK-rank",
    )
    print("{:<16} {:>8} {:>10} {:>11} {:>11} {:>11}".format(*header))
    for row in profiles:
        print(
            f"{row.name:<16} {row.d_tp:>8.4f} {row.trajectory_rank:>10d} "
            f"{row.qfim_rank:>11d} {row.cfim_rank:>11d} {row.qntk_rank:>11d}"
        )

    if args.save:
        output = Path(__file__).resolve().parent / "results"
        csv_path, json_path = save_profiles(output, profiles)
        print(f"\nsaved: {csv_path}")
        print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
