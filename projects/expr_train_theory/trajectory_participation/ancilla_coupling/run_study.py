from pathlib import Path

from .study import run_ancilla_coupling_study, save_results


def main():
    rows, summary, growth_rows, growth_summary = run_ancilla_coupling_study()
    output_dir = Path(__file__).resolve().parent / "results"
    metadata = {
        "kind": "native shared-QNN trajectory/ancilla coupling diagnostic",
        "weights": "untrained random matched parameter prefixes",
        "inputs": "synthetic",
        "interpretation": (
            "depth-wise association between trajectory reach and instantaneous "
            "system-ancilla entanglement; not a causal or trained-model result"
        ),
    }
    save_results(output_dir, rows, summary, growth_rows, growth_summary, metadata=metadata)
    print(f"saved {len(rows)} trajectory rows to {output_dir}")


if __name__ == "__main__":
    main()
