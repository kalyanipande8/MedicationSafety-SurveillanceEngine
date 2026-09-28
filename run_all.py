"""
Orchestrates the full pipeline end to end:
  1. Generate synthetic dispensing data (fixed seed, ground-truth labels).
  2. Compute the 5 documented detection rules + train the composite model.
  3. Evaluate against real held-out ground truth (per-site + aggregate).
  4. Generate real example alert records from the flagged events.
  5. Write output/report.md and 2 real matplotlib figures.

Run: python3 run_all.py
"""
from __future__ import annotations

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_medication_dispensing_data import generate_dataset
from evaluate_detection import run_full_evaluation, DEFAULT_THRESHOLD
from alert_dashboard_stub import generate_alerts

OUTPUT_DIR = "output"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")


def make_tradeoff_figure(sweep_df: pd.DataFrame, chosen_threshold: float, path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep_df["threshold"], sweep_df["sensitivity"], marker="o", label="Sensitivity (recall)")
    ax.plot(sweep_df["threshold"], sweep_df["specificity"], marker="s", label="Specificity")
    ax.axvline(chosen_threshold, color="gray", linestyle="--", alpha=0.7,
               label=f"Chosen threshold = {chosen_threshold}")
    ax.set_xlabel("Composite risk-score threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Sensitivity / Specificity tradeoff vs. detection threshold\n(held-out test split)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_per_site_figure(by_site: pd.DataFrame, path: str):
    by_site = by_site.copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(by_site))
    width = 0.35
    ax.bar([i - width / 2 for i in x], by_site["sensitivity"].fillna(0), width, label="Sensitivity")
    ax.bar([i + width / 2 for i in x], by_site["specificity"].fillna(0), width, label="Specificity")
    ax.set_xticks(list(x))
    ax.set_xticklabels(by_site.index, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Per-site detection performance (held-out test split)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(results: dict, alerts: list, path: str):
    agg = results["aggregate_metrics"]
    by_site = results["per_site_metrics"]
    threshold = results["threshold"]

    lines = []
    lines.append("# Medication Diversion Surveillance -- Evaluation Report\n")
    lines.append(f"Composite risk-score threshold used: **{threshold}**\n")
    lines.append("## Aggregate metrics (held-out test split, real ground truth)\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Sensitivity (recall) | {agg['sensitivity']*100:.1f}% |")
    lines.append(f"| Specificity | {agg['specificity']*100:.1f}% |")
    lines.append(f"| Precision | {agg['precision']*100:.1f}% |")
    lines.append(f"| False positive rate | {agg['false_positive_rate']*100:.1f}% |")
    lines.append(f"| True positives / False negatives | {agg['tp']} / {agg['fn']} |")
    lines.append(f"| True negatives / False positives | {agg['tn']} / {agg['fp']} |")
    lines.append("")
    lines.append("## Per-site metrics\n")
    lines.append(by_site.round(4).to_markdown())
    lines.append("")
    lines.append("## Example generated alerts\n")
    for a in alerts[:5]:
        lines.append(f"- **{a['site']} / {a['provider_id']}** -- event {a['event_id']} "
                      f"({a['medication']}), risk_score={a['risk_score']:.2f}, "
                      f"reasons: {', '.join(a['reasons'])}, action: {a['recommended_action']}")
    lines.append("")
    lines.append("See README.md for full discussion of the sensitivity/specificity tradeoff.\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("[1/5] Generating synthetic dispensing data...")
    df = generate_dataset()
    df.to_csv("data/dispensing_events.csv", index=False)
    print(f"  {len(df):,} events, {df['is_diversion'].sum():,} labeled diversion events "
          f"({df['is_diversion'].mean()*100:.2f}%)")

    print("[2/5] Computing rules + training composite detector, evaluating on held-out split...")
    results = run_full_evaluation(df, threshold=DEFAULT_THRESHOLD)
    agg = results["aggregate_metrics"]
    print(f"  Aggregate sensitivity={agg['sensitivity']*100:.1f}%  "
          f"specificity={agg['specificity']*100:.1f}%  precision={agg['precision']*100:.1f}%")

    print("[3/5] Per-site metrics:")
    print(results["per_site_metrics"].round(4))

    print("[4/5] Generating example structured alerts from flagged test-split events...")
    alerts = generate_alerts(results["test_df"], threshold=results["threshold"])
    print(f"  Generated {len(alerts)} alerts. Example:")
    if alerts:
        print(" ", alerts[0])

    print("[5/5] Writing report + figures...")
    write_report(results, alerts, os.path.join(OUTPUT_DIR, "report.md"))
    make_tradeoff_figure(results["threshold_sweep"], results["threshold"],
                          os.path.join(FIG_DIR, "sensitivity_specificity_tradeoff.png"))
    make_per_site_figure(results["per_site_metrics"],
                          os.path.join(FIG_DIR, "per_site_performance.png"))
    print(f"Done. See {OUTPUT_DIR}/report.md and {FIG_DIR}/")


if __name__ == "__main__":
    main()
