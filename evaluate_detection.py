"""
Evaluate the composite diversion detector against real injected ground-truth
labels on a held-out test split.

Splits by PROVIDER (not by event) so that a provider's events don't leak
across train/test -- this is the honest way to evaluate a surveillance
system meant to catch providers, not just memorize event-level noise.

Computes real sensitivity (recall), specificity, precision, and false
positive rate, per site and in aggregate, at a chosen operating threshold.
Also sweeps thresholds to show the real sensitivity/specificity tradeoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from surveillance.detection_rules import (
    compute_all_rules,
    train_composite_model,
    score_events,
    RULE_COLUMNS,
)

DEFAULT_THRESHOLD = 0.75  # chosen operating point: ~96% sensitivity with much
# improved specificity/precision vs. lower thresholds (see threshold_sweep /
# output/figures/sensitivity_specificity_tradeoff.png for the full tradeoff)


def train_test_split_by_provider(df: pd.DataFrame, test_size=0.4, seed=42):
    """Split by PROVIDER (no provider's events leak across train/test),
    done independently WITHIN each site, and stratified by whether the
    provider is a genuine diverting provider -- so every site keeps at
    least one diverting AND one non-diverting provider on both sides of the
    split. (A plain unstratified group split can, by chance, put both of a
    site's ~2 diverting providers entirely into train, leaving that site
    with zero positive test examples -- not useful for evaluation.)"""
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []

    for site, site_df in df.groupby("site"):
        provider_diverting = (
            site_df.groupby("provider_id")["diverting_provider_ground_truth"].max()
        )
        diverting_providers = provider_diverting[provider_diverting == 1].index.to_numpy()
        clean_providers = provider_diverting[provider_diverting == 0].index.to_numpy()

        def _split(providers):
            providers = providers.copy()
            rng.shuffle(providers)
            n_test = max(1, int(round(len(providers) * test_size))) if len(providers) > 1 else 0
            return providers[n_test:], providers[:n_test]

        train_div, test_div = _split(diverting_providers)
        train_clean, test_clean = _split(clean_providers)

        train_providers = set(train_div) | set(train_clean)
        test_providers = set(test_div) | set(test_clean)

        train_parts.append(site_df[site_df["provider_id"].isin(train_providers)])
        test_parts.append(site_df[site_df["provider_id"].isin(test_providers)])

    return pd.concat(train_parts).copy(), pd.concat(test_parts).copy()


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return tp, fn, tn, fp


def metrics_from_counts(tp, fn, tn, fp):
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "false_positive_rate": fpr,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def evaluate_at_threshold(test_df: pd.DataFrame, threshold: float) -> dict:
    y_true = test_df["is_diversion"].values
    y_pred = (test_df["risk_score"].values >= threshold).astype(int)
    tp, fn, tn, fp = confusion_counts(y_true, y_pred)
    return metrics_from_counts(tp, fn, tn, fp)


def per_site_metrics(test_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for site, grp in test_df.groupby("site"):
        m = evaluate_at_threshold(grp, threshold)
        m["site"] = site
        m["n_events"] = len(grp)
        m["n_diversion"] = int(grp["is_diversion"].sum())
        rows.append(m)
    return pd.DataFrame(rows).set_index("site")


def threshold_sweep(test_df: pd.DataFrame, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for t in thresholds:
        m = evaluate_at_threshold(test_df, t)
        m["threshold"] = t
        rows.append(m)
    return pd.DataFrame(rows)


def run_full_evaluation(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD):
    """Full pipeline: compute rules, provider-level split, train composite
    model on train split, score test split, return metrics + scored df."""
    df_rules = compute_all_rules(df)
    train_df, test_df = train_test_split_by_provider(df_rules)

    model = train_composite_model(train_df)
    test_df = test_df.copy()
    test_df["risk_score"] = score_events(test_df, model)
    train_df = train_df.copy()
    train_df["risk_score"] = score_events(train_df, model)

    agg = evaluate_at_threshold(test_df, threshold)
    by_site = per_site_metrics(test_df, threshold)
    sweep = threshold_sweep(test_df)

    return {
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
        "aggregate_metrics": agg,
        "per_site_metrics": by_site,
        "threshold_sweep": sweep,
        "threshold": threshold,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/dispensing_events.csv")
    results = run_full_evaluation(df, threshold=DEFAULT_THRESHOLD)
    print(f"Threshold = {results['threshold']}")
    print("Aggregate:", results["aggregate_metrics"])
    print("\nPer-site:\n", results["per_site_metrics"])
