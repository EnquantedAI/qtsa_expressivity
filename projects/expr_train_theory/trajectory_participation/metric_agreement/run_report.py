from __future__ import annotations

import argparse
from pathlib import Path

from .agreement import build_agreement_report, save_report, summarize_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    rows = build_agreement_report()
    for row in rows:
        print(
            f"{row.name:18s}  {row.visibility_pattern:24s} "
            f"state-match={row.state_space_agreement!s:5s} "
            f"output-match={row.output_space_agreement!s:5s}"
        )

    summary = summarize_report(rows)
    print("\nsummary")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.save:
        output = Path(__file__).resolve().parent / "results"
        paths = save_report(output, rows)
        print("\nsaved")
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
