from pathlib import Path

from .study import run_real_data_stability, save_results


def main():
    raw, effects, summary, metadata = run_real_data_stability()
    output_dir = Path(__file__).resolve().parent / "results"
    save_results(output_dir, raw, effects, summary, metadata)
    print(f"saved {len(raw)} raw observations and {len(effects)} matched effects to {output_dir}")
    resolved = sum(row["classification"] != "unresolved" for row in summary)
    print(f"resolved stability summaries: {resolved}/{len(summary)}")
    print("weights are untrained; stability refers to architecture diagnostics, not forecasting performance")


if __name__ == "__main__":
    main()
