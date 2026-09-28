"""
Real pytest tests against the real synthetic data + real detector.

Run: pytest -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_medication_dispensing_data import generate_dataset
from surveillance.detection_rules import compute_all_rules, train_composite_model, score_events, RULE_COLUMNS
from evaluate_detection import run_full_evaluation, DEFAULT_THRESHOLD


@pytest.fixture(scope="module")
def df():
    return generate_dataset()


@pytest.fixture(scope="module")
def evaluation(df):
    return run_full_evaluation(df, threshold=DEFAULT_THRESHOLD)


def test_eight_sites_present(df):
    assert df["site"].nunique() == 8


def test_ground_truth_is_genuine_minority_class(df):
    rate = df["is_diversion"].mean()
    assert 0 < rate < 0.15, f"diversion rate {rate:.3f} should be a small minority class"
    assert df["is_diversion"].sum() > 50, "should have enough positive events to evaluate meaningfully"


def test_aggregate_sensitivity_in_reasonable_band(evaluation):
    sens = evaluation["aggregate_metrics"]["sensitivity"]
    assert 0.90 <= sens <= 1.0, f"aggregate sensitivity {sens:.3f} out of expected 90-100% band"


def test_specificity_is_realistic_not_perfect(evaluation):
    # a real high-sensitivity clinical alert system should show a real,
    # non-trivial false-positive cost -- specificity should not be ~100%,
    # which would indicate label leakage or an unrealistically easy task.
    spec = evaluation["aggregate_metrics"]["specificity"]
    assert 0.80 <= spec < 1.0, f"specificity {spec:.3f} looks unrealistic (too perfect or too poor)"


def test_composite_scoring_logic_sanity_check(df):
    """Verify the SCORING LOGIC itself (not just tuning): an event with an
    extreme dispense-vs-waste discrepancy AND after-hours access, from a
    provider with elevated peer-deviation and override behavior, must score
    a higher composite risk than a squeaky-clean normal event -- regardless
    of the trained threshold."""
    df_rules = compute_all_rules(df)
    model = train_composite_model(df_rules)

    normal_event = df_rules.iloc[[0]].copy()
    normal_event["dispensed_qty"] = 10.0
    normal_event["administered_qty"] = 10.0
    normal_event["wasted_qty"] = 0.0
    normal_event["unaccounted_qty"] = 0.0
    normal_event["after_hours"] = 0
    normal_event["witnessed_cosign"] = 1
    normal_event["cabinet_override"] = 0
    for col in RULE_COLUMNS:
        normal_event[col] = 0.0

    suspicious_event = normal_event.copy()
    suspicious_event["dispensed_qty"] = 10.0
    suspicious_event["unaccounted_qty"] = 8.5  # extreme discrepancy
    suspicious_event["after_hours"] = 1
    suspicious_event["witnessed_cosign"] = 0
    suspicious_event["cabinet_override"] = 1
    suspicious_event["rule_waste_discrepancy"] = 0.85
    suspicious_event["rule_peer_deviation"] = 0.9
    suspicious_event["rule_after_hours"] = 0.9
    suspicious_event["rule_missing_cosign"] = 1.0
    suspicious_event["rule_override_pattern"] = 0.9

    combined = pd.concat([normal_event, suspicious_event])
    scores = score_events(combined, model)

    assert scores.iloc[1] > scores.iloc[0], (
        "an event with extreme discrepancy + after-hours + missing cosign + "
        "override should score strictly higher risk than a clean normal event"
    )
    assert scores.iloc[1] > 0.5, "the clearly suspicious synthetic event should cross a moderate risk threshold"


def test_per_site_metrics_cover_all_sites(evaluation, df):
    by_site = evaluation["per_site_metrics"]
    assert set(by_site.index) == set(df["site"].unique())


def test_rule_columns_bounded_0_1(df):
    df_rules = compute_all_rules(df)
    for col in RULE_COLUMNS:
        assert df_rules[col].min() >= -1e-9
        assert df_rules[col].max() <= 1 + 1e-9
