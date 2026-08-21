from __future__ import annotations

from .cases import run_boundary_cases


def main():
    for row in run_boundary_cases():
        print(row)


if __name__ == "__main__":
    main()
