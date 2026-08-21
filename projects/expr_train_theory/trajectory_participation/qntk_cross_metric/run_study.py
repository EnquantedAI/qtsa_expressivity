from __future__ import annotations

from pathlib import Path

from .study import run_study, save_results


def main():
    rows = run_study()
    output_dir = Path(__file__).resolve().parent / "results"
    save_results(output_dir, rows, {"model": "one-qubit RY/RZ toy model"})

    for row in rows:
        print(
            f"{row['case']}: dTP={row['d_tp_mean']:.4f}, "
            f"QNTK rank={row['qntk_rank']}, "
            f"QNTK eff.rank={row['qntk_effective_rank']:.4f}, "
            f"trace={row['qntk_trace']:.4f}"
        )


if __name__ == "__main__":
    main()
