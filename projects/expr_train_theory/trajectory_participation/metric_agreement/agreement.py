from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

from projects.expr_train_theory.trajectory_participation.metric_profile.profile import (
    MetricProfile,
    default_profiles,
)


@dataclass(frozen=True)
class AgreementRow:
    name: str
    trajectory_active: bool
    qfim_active: bool
    cfim_active: bool
    qntk_active: bool
    state_space_agreement: bool
    output_space_agreement: bool
    all_active_agreement: bool
    qfim_cfim_rank_gap: int
    qfim_qntk_rank_gap: int
    visibility_pattern: str


def _visibility_pattern(tp: bool, qfim: bool, cfim: bool, qntk: bool) -> str:
    labels = []
    if tp:
        labels.append("dTP")
    if qfim:
        labels.append("QFIM")
    if cfim:
        labels.append("CFIM")
    if qntk:
        labels.append("QNTK")
    return "+".join(labels) if labels else "none"


def compare_profile(profile: MetricProfile, *, d_tp_tol: float = 1e-8) -> AgreementRow:
    trajectory_active = bool(profile.d_tp > 1.0 + d_tp_tol or profile.trajectory_rank > 1)
    qfim_active = bool(profile.qfim_rank > 0)
    cfim_active = bool(profile.cfim_rank > 0)
    qntk_active = bool(profile.qntk_rank > 0)

    return AgreementRow(
        name=profile.name,
        trajectory_active=trajectory_active,
        qfim_active=qfim_active,
        cfim_active=cfim_active,
        qntk_active=qntk_active,
        state_space_agreement=trajectory_active == qfim_active,
        output_space_agreement=cfim_active == qntk_active,
        all_active_agreement=len({trajectory_active, qfim_active, cfim_active, qntk_active}) == 1,
        qfim_cfim_rank_gap=int(profile.qfim_rank - profile.cfim_rank),
        qfim_qntk_rank_gap=int(profile.qfim_rank - profile.qntk_rank),
        visibility_pattern=_visibility_pattern(
            trajectory_active,
            qfim_active,
            cfim_active,
            qntk_active,
        ),
    )


def build_agreement_report(profiles=None):
    if profiles is None:
        profiles = default_profiles()
    return [compare_profile(profile) for profile in profiles]


def summarize_report(rows):
    rows = list(rows)
    if not rows:
        return {
            "case_count": 0,
            "state_space_agreement_fraction": 0.0,
            "output_space_agreement_fraction": 0.0,
            "all_active_agreement_fraction": 0.0,
            "patterns": {},
        }

    patterns = {}
    for row in rows:
        patterns[row.visibility_pattern] = patterns.get(row.visibility_pattern, 0) + 1

    n = float(len(rows))
    return {
        "case_count": len(rows),
        "state_space_agreement_fraction": sum(row.state_space_agreement for row in rows) / n,
        "output_space_agreement_fraction": sum(row.output_space_agreement for row in rows) / n,
        "all_active_agreement_fraction": sum(row.all_active_agreement for row in rows) / n,
        "patterns": patterns,
    }


def save_report(output_dir, rows):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(rows)

    csv_path = output / "metric_agreement.csv"
    json_path = output / "metric_agreement_summary.json"

    records = [asdict(row) for row in rows]
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    else:
        csv_path.write_text("", encoding="utf-8")

    summary = summarize_report(rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path
