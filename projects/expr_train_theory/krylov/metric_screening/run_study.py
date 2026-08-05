from __future__ import annotations

from pathlib import Path

from .screening import ScreeningConfig, run_screening
from ..circuit_ensemble.study import EnsembleConfig


def main() -> None:
    config = ScreeningConfig(
        ensemble=EnsembleConfig(
            n_qubits=3,
            depths=(1, 2, 3, 4, 5),
            topologies=("none", "linear", "ring", "full"),
            samples=20,
            seed=2026,
        )
    )
    output = Path(__file__).resolve().parent / "results"
    _, rows = run_screening(config, output)

    print("metric screening")
    for row in rows:
        print(
            f"{row['metric']:<36} "
            f"depth eta2={row['depth_eta_squared']:.3f}  "
            f"topology eta2={row['topology_eta_squared']:.3f}  "
            f"max |rho|={row['max_abs_spearman']:.3f}"
        )
        print(f"  {row['note']}")
    print(f"\nresults: {output}")


if __name__ == "__main__":
    main()
