from pathlib import Path

from .study import run_reference_cases


def main():
    output = Path(__file__).with_name("results")
    results = run_reference_cases(output)
    for row in results:
        print(
            f"{row.name:22s}  dTP={row.d_tp:.4f}  "
            f"traj-rank={row.trajectory_rank}  "
            f"QFIM-rank={row.qfim_rank}/{row.parameter_count}  "
            f"CFIM-rank={row.cfim_rank}/{row.parameter_count}"
        )


if __name__ == "__main__":
    main()
