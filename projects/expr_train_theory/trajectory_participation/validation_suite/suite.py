from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..boundary_validation.cases import run_boundary_cases
from ..directional_validation.analysis import run_reference_study
from ..perturbation_stability.study import (
    snapshot_perturbation_study,
    summarise_rows,
)
from ..sampling_sensitivity.study import duplication_study, sampling_density_study
from ..snapshots import trajectory_snapshots


def _hard_check(name, value, expected, tolerance=1e-8):
    error = abs(float(value) - float(expected))
    return {
        "name": name,
        "kind": "hard",
        "status": "pass" if error <= tolerance else "fail",
        "value": float(value),
        "expected": float(expected),
        "error": float(error),
        "tolerance": float(tolerance),
    }


def boundary_checks():
    cases = {row["case"]: row for row in run_boundary_cases()}
    checks = [
        _hard_check(
            "collapsed trajectory dTP",
            cases["collapsed_trajectory"]["d_tp"],
            1.0,
        ),
        _hard_check(
            "orthogonal trajectory dTP",
            cases["orthogonal_trajectory"]["d_tp"],
            cases["orthogonal_trajectory"]["expected_d_tp"],
        ),
        _hard_check(
            "constant-output QNTK norm",
            cases["constant_output_qntk"]["kernel_norm"],
            0.0,
            tolerance=1e-7,
        ),
        _hard_check(
            "linear QNTK Jacobian error",
            cases["linear_qntk"]["jacobian_error"],
            0.0,
            tolerance=1e-6,
        ),
        _hard_check(
            "linear QNTK kernel error",
            cases["linear_qntk"]["kernel_error"],
            0.0,
            tolerance=1e-6,
        ),
    ]

    redundant = cases["redundant_parameter_qntk"]
    checks.append(
        {
            "name": "redundant-parameter QNTK rank",
            "kind": "hard",
            "status": "pass"
            if redundant["jacobian_rank"] == 1 and redundant["qntk_rank"] == 1
            else "fail",
            "value": int(redundant["qntk_rank"]),
            "expected": 1,
            "jacobian_rank": int(redundant["jacobian_rank"]),
            "parameter_count": int(redundant["parameter_count"]),
        }
    )
    return checks


def sampling_checks():
    duplicate = duplication_study(duplicate_count=8)
    density = sampling_density_study(sample_counts=(3, 5, 9, 17, 33))
    checks = [
        _hard_check("two orthogonal states", duplicate["base"], 2.0),
        _hard_check(
            "weighted duplicate correction",
            duplicate["duplicated_weighted"],
            duplicate["base"],
        ),
        {
            "name": "equal-weight duplicate sensitivity",
            "kind": "diagnostic",
            "status": "info",
            "base": float(duplicate["base"]),
            "duplicated": float(duplicate["duplicated_equal"]),
            "delta": float(duplicate["duplicated_equal"] - duplicate["base"]),
        },
        {
            "name": "sampling-density convergence",
            "kind": "diagnostic",
            "status": "info",
            "first_equal": float(density[0]["d_tp_equal"]),
            "last_equal": float(density[-1]["d_tp_equal"]),
            "first_weighted": float(density[0]["d_tp_trapezoidal"]),
            "last_weighted": float(density[-1]["d_tp_trapezoidal"]),
        },
    ]
    return checks


def perturbation_checks(seed=2026):
    rng = np.random.default_rng(seed)
    features = rng.uniform(-1.0, 1.0, size=2)
    parameters = rng.uniform(-0.5, 0.5, size=(3, 2, 3))
    states = trajectory_snapshots(features, parameters, n_qubits=2, entangle=True)

    rows = snapshot_perturbation_study(
        states,
        scales=(0.0, 1e-4, 1e-3, 1e-2),
        repeats=8,
        seed=seed,
    )
    summary = summarise_rows(rows)
    by_scale = {row["scale"]: row for row in summary}

    zero = by_scale[0.0]
    small = by_scale[1e-4]
    large = by_scale[1e-2]
    return [
        _hard_check("zero perturbation dTP delta", zero["mean_abs_delta_d_tp"], 0.0),
        _hard_check("zero perturbation infidelity", zero["mean_infidelity"], 0.0),
        {
            "name": "perturbation response",
            "kind": "diagnostic",
            "status": "info",
            "small_scale": 1e-4,
            "small_mean_abs_delta_d_tp": float(small["mean_abs_delta_d_tp"]),
            "large_scale": 1e-2,
            "large_mean_abs_delta_d_tp": float(large["mean_abs_delta_d_tp"]),
            "larger_at_larger_scale": bool(
                large["mean_abs_delta_d_tp"] >= small["mean_abs_delta_d_tp"]
            ),
        },
    ]


def directional_diagnostics(seed=2026):
    _, summaries, paired = run_reference_study(
        layers=(1, 2, 3, 4), qubits=(1, 2), samples=8, seed=seed
    )
    selected = []
    for row in summaries:
        if row["metric"] == "d_tp_normalized":
            selected.append(
                {
                    "name": f"depth trend q={row['n_qubits']} reupload={row['reupload']}",
                    "kind": "diagnostic",
                    "status": "info",
                    "spearman": row["spearman"],
                    "positive_steps": row["positive_steps"],
                    "negative_steps": row["negative_steps"],
                    "flat_steps": row["flat_steps"],
                    "delta": row["delta"],
                }
            )

    deltas = np.array([row["d_tp_delta_reupload"] for row in paired], dtype=float)
    selected.append(
        {
            "name": "paired reuploading effect",
            "kind": "diagnostic",
            "status": "info",
            "mean_delta_d_tp": float(np.mean(deltas)),
            "median_delta_d_tp": float(np.median(deltas)),
            "positive_fraction": float(np.mean(deltas > 0.0)),
        }
    )
    return selected


def run_validation_suite(seed=2026):
    checks = []
    checks.extend(boundary_checks())
    checks.extend(sampling_checks())
    checks.extend(perturbation_checks(seed=seed))
    checks.extend(directional_diagnostics(seed=seed))

    hard = [row for row in checks if row["kind"] == "hard"]
    return {
        "summary": {
            "hard_checks": len(hard),
            "hard_passed": sum(row["status"] == "pass" for row in hard),
            "hard_failed": sum(row["status"] == "fail" for row in hard),
            "diagnostics": sum(row["kind"] == "diagnostic" for row in checks),
            "seed": int(seed),
        },
        "checks": checks,
    }


def save_report(output_dir, report):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    lines = [
        "# Trajectory participation validation",
        "",
        f"Hard checks: {report['summary']['hard_passed']}/{report['summary']['hard_checks']} passed.",
        "",
        "The trend rows below are diagnostics, not pass/fail conditions.",
        "",
    ]
    for row in report["checks"]:
        if row["kind"] == "hard":
            lines.append(f"- **{row['status'].upper()}** — {row['name']}")
        else:
            lines.append(f"- **INFO** — {row['name']}")
    lines.append("")
    (output_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
