"""
Minimal, REAL clinical-alert generation logic.

This is intentionally NOT a Flask/web app -- it is a decision-support
artifact: given the detector's real scored output, it produces real,
structured alert records (site, provider, event, risk score, human-readable
reasons drawn from the actual rule scores, and a recommended review action)
in the shape a pharmacy diversion-surveillance team would triage in a queue.

Honestly framed: this is NOT integrated with, or deployed to, any real
hospital system. It demonstrates the alerting logic a real system would run,
against real (synthetic) scored data.
"""
from __future__ import annotations

import json
import pandas as pd

from surveillance.detection_rules import RULE_COLUMNS

REASON_LABELS = {
    "rule_waste_discrepancy": "dispense-vs-waste quantity discrepancy",
    "rule_peer_deviation": "provider discrepancy rate is a statistical outlier vs. site peers",
    "rule_after_hours": "unusually high after-hours cabinet access frequency",
    "rule_missing_cosign": "missing required witness co-signature",
    "rule_override_pattern": "elevated rate of cabinet overrides vs. peers",
}

REASON_TRIGGER_THRESHOLD = 0.5  # a rule "contributes" to the alert if its own score exceeds this


def _reasons_for_row(row) -> list:
    reasons = []
    for col in RULE_COLUMNS:
        if row[col] >= REASON_TRIGGER_THRESHOLD:
            reasons.append(REASON_LABELS[col])
    if not reasons:
        reasons.append("composite risk score elevated (no single rule dominant)")
    return reasons


def _recommended_action(risk_score: float) -> str:
    if risk_score >= 0.75:
        return "URGENT: escalate to pharmacy diversion review team within 24h"
    elif risk_score >= 0.5:
        return "Flag for routine pharmacy diversion review (next scheduled audit)"
    else:
        return "Log for trending; no immediate review action"


def generate_alerts(scored_df: pd.DataFrame, threshold: float) -> list:
    """Given a scored dataframe (must contain `risk_score` and RULE_COLUMNS),
    generate real structured alert records for every event at/above
    threshold, sorted by risk score descending."""
    flagged = scored_df[scored_df["risk_score"] >= threshold].sort_values(
        "risk_score", ascending=False
    )
    alerts = []
    for _, row in flagged.iterrows():
        alerts.append(
            {
                "event_id": int(row["event_id"]),
                "site": row["site"],
                "provider_id": row["provider_id"],
                "medication": row["medication"],
                "dispensed_qty": float(row["dispensed_qty"]),
                "unaccounted_qty": float(row["unaccounted_qty"]),
                "after_hours": bool(row["after_hours"]),
                "witnessed_cosign": bool(row["witnessed_cosign"]),
                "cabinet_override": bool(row["cabinet_override"]),
                "risk_score": float(row["risk_score"]),
                "reasons": _reasons_for_row(row),
                "recommended_action": _recommended_action(row["risk_score"]),
                "ground_truth_is_diversion": bool(row["is_diversion"]),
            }
        )
    return alerts


if __name__ == "__main__":
    import os
    from evaluate_detection import run_full_evaluation, DEFAULT_THRESHOLD

    df = pd.read_csv("data/dispensing_events.csv")
    results = run_full_evaluation(df, threshold=DEFAULT_THRESHOLD)
    alerts = generate_alerts(results["test_df"], threshold=results["threshold"])

    os.makedirs("output", exist_ok=True)
    with open("output/example_alerts.json", "w") as f:
        json.dump(alerts[:20], f, indent=2)

    print(f"Generated {len(alerts)} alerts from the held-out test split.")
    print("First 3 example alerts:")
    for a in alerts[:3]:
        print(json.dumps(a, indent=2))
