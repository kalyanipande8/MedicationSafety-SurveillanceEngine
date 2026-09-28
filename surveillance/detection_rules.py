"""
Real, individually-documented medication diversion detection rules.

Each rule below is a small, human-auditable function that turns raw
dispensing-event fields into a bounded [0, 1] "rule score" for a single
event. Rules are combined into one composite risk score per event using a
logistic-regression combiner trained on a labeled training split (see
`train_composite_model` / `score_events`). This mirrors how a real clinical
surveillance program typically works: individually explainable rules
("why was this flagged?") feeding a single ranked risk score for pharmacy
review, rather than a single opaque black-box model.

Rules implemented:
  1. dispense_waste_discrepancy_rule  - unaccounted quantity vs. dispensed qty
  2. peer_deviation_rule              - z-score of a provider's per-event
                                         discrepancy/override behavior vs.
                                         same-site peers
  3. after_hours_frequency_rule       - a provider's share of after-hours
                                         dispensing vs. site baseline
  4. missing_cosign_rule              - lack of a required witness co-sign,
                                         especially combined with an override
  5. override_pattern_rule            - repeated cabinet overrides by the
                                         same provider (rate vs. site peers)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

RULE_COLUMNS = [
    "rule_waste_discrepancy",
    "rule_peer_deviation",
    "rule_after_hours",
    "rule_missing_cosign",
    "rule_override_pattern",
]


def _zscore_clip(x: pd.Series) -> pd.Series:
    mu, sd = x.mean(), x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rule_dispense_waste_discrepancy(df: pd.DataFrame) -> pd.Series:
    """Rule 1: fraction of the dispensed quantity that is unaccounted for
    (dispensed - administered - wasted). A real diversion signature: the
    cabinet says X was dispensed but only Y was actually given/wasted."""
    frac_unaccounted = (df["unaccounted_qty"] / df["dispensed_qty"].replace(0, np.nan)).fillna(0)
    frac_unaccounted = frac_unaccounted.clip(lower=0)
    return frac_unaccounted.clip(upper=1.0)


def rule_peer_deviation(df: pd.DataFrame) -> pd.Series:
    """Rule 2: z-score (within site) of each PROVIDER's mean unaccounted
    fraction, broadcast back to their events. Flags a provider whose overall
    dispensing behavior is a statistical outlier vs. same-site peers, not
    just a single noisy event."""
    provider_mean = df.groupby(["site", "provider_id"])["unaccounted_qty"].transform("mean")
    dispensed_mean = df.groupby(["site", "provider_id"])["dispensed_qty"].transform("mean")
    provider_frac = (provider_mean / dispensed_mean.replace(0, np.nan)).fillna(0)
    # compute z-score of provider_frac within each site
    out = pd.Series(index=df.index, dtype=float)
    tmp = provider_frac.copy()
    tmp.name = "provider_frac"
    joined = pd.concat([df["site"], tmp], axis=1)
    for site, grp in joined.groupby("site"):
        z_site = _zscore_clip(grp["provider_frac"])
        out.loc[grp.index] = z_site
    score = _sigmoid((out - 1.0) * 1.5)  # centers "flag" around z > ~1
    return score


def rule_after_hours_frequency(df: pd.DataFrame) -> pd.Series:
    """Rule 3: a provider's after-hours access RATE vs. their site's overall
    after-hours rate, expressed as a z-score-based score. A provider who
    disproportionately accesses controlled substances overnight relative to
    peers is a classic diversion surveillance signal."""
    provider_rate = df.groupby(["site", "provider_id"])["after_hours"].transform("mean")
    out = pd.Series(index=df.index, dtype=float)
    tmp = provider_rate.copy()
    tmp.name = "provider_rate"
    joined = pd.concat([df["site"], tmp], axis=1)
    for site, grp in joined.groupby("site"):
        z_site = _zscore_clip(grp["provider_rate"])
        out.loc[grp.index] = z_site
    score = _sigmoid((out - 1.0) * 1.3)
    return score


def rule_missing_cosign(df: pd.DataFrame) -> pd.Series:
    """Rule 4: missing witness co-signature, weighted higher when combined
    with a cabinet override (an override without a co-sign is a much
    stronger signal than a routine dispense without a co-sign)."""
    base = (1 - df["witnessed_cosign"]).astype(float) * 0.4
    combo = ((1 - df["witnessed_cosign"]) & (df["cabinet_override"] == 1)).astype(float) * 0.6
    return (base + combo).clip(upper=1.0)


def rule_override_pattern(df: pd.DataFrame) -> pd.Series:
    """Rule 5: a provider's cabinet-override RATE vs. same-site peers
    (z-score based). Repeated overrides (bypassing normal dispensing
    workflow) concentrated in one provider is a documented diversion
    pattern independent of any single event's discrepancy."""
    provider_rate = df.groupby(["site", "provider_id"])["cabinet_override"].transform("mean")
    out = pd.Series(index=df.index, dtype=float)
    tmp = provider_rate.copy()
    tmp.name = "provider_rate"
    joined = pd.concat([df["site"], tmp], axis=1)
    for site, grp in joined.groupby("site"):
        z_site = _zscore_clip(grp["provider_rate"])
        out.loc[grp.index] = z_site
    score = _sigmoid((out - 1.0) * 1.3)
    return score


def compute_all_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 5 rule scores and attach as columns; returns a copy."""
    out = df.copy()
    out["rule_waste_discrepancy"] = rule_dispense_waste_discrepancy(out)
    out["rule_peer_deviation"] = rule_peer_deviation(out)
    out["rule_after_hours"] = rule_after_hours_frequency(out)
    out["rule_missing_cosign"] = rule_missing_cosign(out)
    out["rule_override_pattern"] = rule_override_pattern(out)
    return out


def train_composite_model(train_df: pd.DataFrame) -> LogisticRegression:
    """Train a logistic-regression combiner over the 5 rule scores using the
    real ground-truth `is_diversion` labels on the TRAINING split only."""
    X = train_df[RULE_COLUMNS].values
    y = train_df["is_diversion"].values
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X, y)
    return model


def score_events(df_with_rules: pd.DataFrame, model: LogisticRegression) -> pd.Series:
    """Apply a trained composite model to produce a risk score in [0, 1]
    per event."""
    X = df_with_rules[RULE_COLUMNS].values
    return pd.Series(model.predict_proba(X)[:, 1], index=df_with_rules.index)
