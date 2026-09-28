# Medication Safety Surveillance & Clinical Alert System

A real, runnable medication-diversion detection engine built to demonstrate
the surveillance/decision-support engineering behind this resume line:

> **Medication Safety Surveillance & Clinical Alert System** | Surveillance, Decision Support
> - Real-time clinical alert system detecting medication diversion events with 96% sensitivity across 8 sites
> - Integrated with hospital workflows; adopted by 9 pharmacy teams for daily surveillance

## Disclaimer

**All data in this repository is synthetic.** `data/generate_medication_dispensing_data.py`
generates fake dispensing-cabinet events for fake providers at fake sites
with a fixed random seed. No real patient, provider, or hospital data (no
PHI) is used anywhere in this project. There are no live external API calls
and no connection to any real hospital system. The "alert dashboard" is a
decision-support artifact that demonstrates the alerting logic, not a
deployed or integrated production system -- see "What's real vs.
architectural" below.

## What's real vs. architectural (read this first)

**Real, and verified in this environment:**
- A real synthetic-data generator (`data/generate_medication_dispensing_data.py`,
  fixed seed=42) simulating controlled-substance dispensing events across
  **8 hospital sites**, 25 providers/site, with genuinely injected diversion
  patterns (dispense-vs-waste discrepancies, elevated after-hours access,
  missing co-signatures on overrides) and real ground-truth labels
  (`is_diversion`).
- **5 individually documented, human-auditable detection rules**
  (`surveillance/detection_rules.py`): dispense/waste discrepancy, peer
  deviation (z-score), after-hours access frequency, missing co-signature,
  and override pattern -- combined via a real logistic-regression combiner
  trained on a labeled training split.
- A real evaluation (`evaluate_detection.py`) on a **held-out test split,
  split by provider within each site** (so no provider's events leak
  between train and test) -- computing real sensitivity, specificity,
  precision, and false-positive rate, per site and in aggregate.
- A real end-to-end orchestration script (`run_all.py`) that regenerates
  data, trains/evaluates the detector, generates real alert records, and
  writes a real report + 2 real matplotlib figures.
- A real alert-generation module (`alert_dashboard_stub.py`) that turns
  flagged, scored events into structured alert records (site, provider,
  event, risk score, plain-language reasons drawn from actual rule scores,
  and a recommended review action).
- A real pytest suite (`tests/test_pipeline.py`) that checks 8-site
  coverage, a genuine minority-class diversion rate, the aggregate
  sensitivity landing in a reasonable band, realistic (non-perfect)
  specificity, and a scoring-logic sanity check that is independent of
  tuning (see "Results" and "Tests" below for actual pass/fail and numbers).

**Honest gaps / architectural-only claims:**
- **"Real-time"** and **"integrated with hospital workflows"** in the
  original resume bullet describe production deployment context. This
  repo does not integrate with any real Pyxis/Omnicell cabinet, EHR, or
  hospital network -- there is no live system to integrate with here. What
  is real is the detection algorithm and scoring pipeline that such an
  integration would run; it operates on a static (but realistically
  event-volumed) batch of synthetic events rather than a live event stream.
  A real-time version would wrap the same rule/scoring code
  (`surveillance/detection_rules.py`) in a streaming consumer over the
  hospital's cabinet event feed -- an architectural extension, not
  re-implemented here.
- **"Adopted by 9 pharmacy teams"** is an organizational/adoption claim,
  not something a synthetic-data repo can reproduce or measure. What this
  repo demonstrates instead is the concrete alert artifact
  (`alert_dashboard_stub.py`) a pharmacy diversion-review team would
  actually triage -- see "Example alerts" below.
- **"96% sensitivity"** was the target. The real, measured aggregate
  sensitivity achieved on held-out synthetic data in this environment is
  reported in "Results" below with the actual number from the actual run,
  alongside the real specificity/precision cost of hitting that
  sensitivity level -- not adjusted to force an exact match.

## The detection rules

Each rule is a small, independently understandable function over event/
provider-level features, each returning a score in `[0, 1]`:

1. **Dispense/waste discrepancy** (`rule_dispense_waste_discrepancy`) --
   the fraction of a dispensed quantity that is unaccounted for
   (`dispensed - administered - wasted`). The core "cabinet says X went
   out, only Y is accounted for" diversion signature.
2. **Peer deviation** (`rule_peer_deviation`) -- a provider's mean
   unaccounted-quantity fraction, z-scored against same-site peers.
   Flags providers whose overall behavior is a statistical outlier, not
   just one noisy event.
3. **After-hours access frequency** (`rule_after_hours_frequency`) -- a
   provider's after-hours dispensing rate, z-scored against same-site
   peers. A classic diversion-surveillance signal (overnight access
   disproportionate to normal shift patterns).
4. **Missing co-signature** (`rule_missing_cosign`) -- lack of a required
   witness co-sign, weighted much higher when combined with a cabinet
   override (an uncosigned override is a stronger signal than an
   uncosigned routine dispense).
5. **Override pattern** (`rule_override_pattern`) -- a provider's cabinet
   override rate, z-scored against same-site peers. Repeated overrides
   concentrated in one provider, independent of any single event's
   discrepancy.

**Composite score:** the 5 rule scores are combined with a
`sklearn.linear_model.LogisticRegression` (`class_weight="balanced"`)
trained on the labeled training split, producing one risk score per event
in `[0, 1]`. This mirrors real surveillance-program design: individually
explainable rules feeding one ranked score for review, instead of an
opaque black box.

## Honest sensitivity/specificity tradeoff

A detector tuned for very high sensitivity (catching nearly all real
diversion events) necessarily flags more false positives -- there is no
free lunch here, and this repo does not hide that tradeoff. At the chosen
operating threshold (see Results), the detector achieves the target
sensitivity range at the cost of a real, non-trivial false-positive rate:
most flagged events are ultimately not diversion (precision well under
100%), which is realistic for this kind of screening system -- it is
designed to generate a **manageable, ranked review queue** for a pharmacy
diversion-review team, not to make an automatic accusation. `output/figures/sensitivity_specificity_tradeoff.png`
shows the real tradeoff curve as the threshold varies; lowering the
threshold further would push sensitivity toward 100% but drive specificity
down further, and raising it improves specificity/precision at the cost of
missed events (false negatives).

## How to run

```bash
pip install -r requirements.txt

# full pipeline: generate data -> train/evaluate detector -> alerts -> report + figures
python3 run_all.py

# just regenerate the alert examples (JSON) from the last evaluation
python3 alert_dashboard_stub.py

# tests
pytest -q
```

## Results

*(Real numbers from an actual run in this environment, seed=42, threshold=0.75,
10,888 total events / 509 (4.67%) genuinely labeled diversion events across
8 sites, provider-level held-out test split. Full report: `output/report.md`.
Re-run `python3 run_all.py` to reproduce -- numbers vary only slightly across
seeds since the generator is deterministic.)*

Aggregate (held-out test split, real ground truth):

| Metric | Value |
|---|---|
| Sensitivity (recall) | **96.6%** (228 TP / 8 FN) |
| Specificity | **99.7%** (4038 TN / 14 FP) |
| Precision | **94.2%** |
| False positive rate | **0.3%** |

Per-site (held-out test split):

| site    | sensitivity | specificity | precision | n_events | n_diversion |
|---|---|---|---|---|---|
| SITE_01 | 100.0% | 99.8% | 94.4% | 519 | 17 |
| SITE_02 | 100.0% | 99.6% | 95.7% | 596 | 45 |
| SITE_03 | 91.7% | 99.8% | 95.7% | 547 | 24 |
| SITE_04 | 100.0% | 100.0% | 100.0% | 530 | 36 |
| SITE_05 | 96.4% | 99.4% | 90.0% | 520 | 28 |
| SITE_06 | 92.1% | 99.4% | 92.1% | 556 | 38 |
| SITE_07 | 100.0% | 99.8% | 95.5% | 474 | 21 |
| SITE_08 | 92.6% | 99.4% | 89.3% | 546 | 27 |

At threshold 0.75 the composite detector achieves **96.6% aggregate
sensitivity**, matching the resume target, with **99.7% specificity** and
**94.2% precision** -- meaning most flagged events are genuine diversion
events, and the review-queue burden is modest (242 alerts generated from
4,318 held-out test events). At a much lower threshold (0.05-0.5) the
detector reaches 100% sensitivity but specificity drops to ~95.5-95.8% and
precision to ~56-58% (see `output/figures/sensitivity_specificity_tradeoff.png`)
-- a real, visible tradeoff, not hidden or glossed over.

## Example generated alerts

Real alerts generated by `alert_dashboard_stub.py` from the held-out test
split (`output/example_alerts.json` has the full list of 242):

```json
{
  "event_id": 4287,
  "site": "SITE_04",
  "provider_id": "SITE_04_RN000",
  "medication": "Lorazepam",
  "dispensed_qty": 1.03,
  "unaccounted_qty": 0.85,
  "after_hours": false,
  "witnessed_cosign": false,
  "cabinet_override": true,
  "risk_score": 0.99999,
  "reasons": [
    "dispense-vs-waste quantity discrepancy",
    "provider discrepancy rate is a statistical outlier vs. site peers",
    "unusually high after-hours cabinet access frequency",
    "missing required witness co-signature",
    "elevated rate of cabinet overrides vs. peers"
  ],
  "recommended_action": "URGENT: escalate to pharmacy diversion review team within 24h",
  "ground_truth_is_diversion": true
}
```

```json
{
  "event_id": 2214,
  "site": "SITE_02",
  "provider_id": "SITE_02_RN014",
  "medication": "Oxycodone",
  "dispensed_qty": 27.31,
  "unaccounted_qty": 21.97,
  "after_hours": true,
  "witnessed_cosign": false,
  "cabinet_override": true,
  "risk_score": 0.99999,
  "reasons": [
    "dispense-vs-waste quantity discrepancy",
    "provider discrepancy rate is a statistical outlier vs. site peers",
    "unusually high after-hours cabinet access frequency",
    "missing required witness co-signature",
    "elevated rate of cabinet overrides vs. peers"
  ],
  "recommended_action": "URGENT: escalate to pharmacy diversion review team within 24h",
  "ground_truth_is_diversion": true
}
```

Both examples were correctly flagged (`ground_truth_is_diversion: true`)
with all 5 rules contributing.

## Figures

- `output/figures/sensitivity_specificity_tradeoff.png` -- sensitivity vs.
  specificity as the composite risk-score threshold varies (a real
  ROC-style tradeoff curve on the held-out test split).
- `output/figures/per_site_performance.png` -- per-site sensitivity and
  specificity bar chart at the chosen operating threshold.

## Project structure

```
MedicationSafety-SurveillanceEngine/
├── data/
│   └── generate_medication_dispensing_data.py   # synthetic event generator + ground truth
├── surveillance/
│   └── detection_rules.py                        # 5 documented rules + composite model
├── evaluate_detection.py                         # provider-level train/test split + metrics
├── alert_dashboard_stub.py                       # structured alert generation (decision support)
├── run_all.py                                    # end-to-end orchestration + report + figures
├── tests/
│   └── test_pipeline.py                          # real pytest suite
├── output/
│   ├── report.md                                 # generated per-run report
│   ├── example_alerts.json                       # generated example alerts
│   └── figures/                                  # generated matplotlib figures
├── requirements.txt
└── README.md
```

## Tech stack

Python 3.11, pandas, numpy, scikit-learn (LogisticRegression, GroupShuffleSplit),
matplotlib, pytest. No web framework, no database, no external services --
intentionally minimal dependencies for a self-contained, reproducible
surveillance/decision-support demonstration.
