# Medication Diversion Surveillance -- Evaluation Report

Composite risk-score threshold used: **0.75**

## Aggregate metrics (held-out test split, real ground truth)

| Metric | Value |
|---|---|
| Sensitivity (recall) | 96.6% |
| Specificity | 99.7% |
| Precision | 94.2% |
| False positive rate | 0.3% |
| True positives / False negatives | 228 / 8 |
| True negatives / False positives | 4038 / 14 |

## Per-site metrics

| site    |   sensitivity |   specificity |   precision |   false_positive_rate |   tp |   fn |   tn |   fp |   n_events |   n_diversion |
|:--------|--------------:|--------------:|------------:|----------------------:|-----:|-----:|-----:|-----:|-----------:|--------------:|
| SITE_01 |        1      |        0.998  |      0.9444 |                0.002  |   17 |    0 |  501 |    1 |        519 |            17 |
| SITE_02 |        1      |        0.9964 |      0.9574 |                0.0036 |   45 |    0 |  549 |    2 |        596 |            45 |
| SITE_03 |        0.9167 |        0.9981 |      0.9565 |                0.0019 |   22 |    2 |  522 |    1 |        547 |            24 |
| SITE_04 |        1      |        1      |      1      |                0      |   36 |    0 |  494 |    0 |        530 |            36 |
| SITE_05 |        0.9643 |        0.9939 |      0.9    |                0.0061 |   27 |    1 |  489 |    3 |        520 |            28 |
| SITE_06 |        0.9211 |        0.9942 |      0.9211 |                0.0058 |   35 |    3 |  515 |    3 |        556 |            38 |
| SITE_07 |        1      |        0.9978 |      0.9545 |                0.0022 |   21 |    0 |  452 |    1 |        474 |            21 |
| SITE_08 |        0.9259 |        0.9942 |      0.8929 |                0.0058 |   25 |    2 |  516 |    3 |        546 |            27 |

## Example generated alerts

- **SITE_04 / SITE_04_RN000** -- event 4287 (Lorazepam), risk_score=1.00, reasons: dispense-vs-waste quantity discrepancy, provider discrepancy rate is a statistical outlier vs. site peers, unusually high after-hours cabinet access frequency, missing required witness co-signature, elevated rate of cabinet overrides vs. peers, action: URGENT: escalate to pharmacy diversion review team within 24h
- **SITE_08 / SITE_08_RN024** -- event 10860 (Lorazepam), risk_score=1.00, reasons: dispense-vs-waste quantity discrepancy, provider discrepancy rate is a statistical outlier vs. site peers, unusually high after-hours cabinet access frequency, missing required witness co-signature, elevated rate of cabinet overrides vs. peers, action: URGENT: escalate to pharmacy diversion review team within 24h
- **SITE_02 / SITE_02_RN014** -- event 2214 (Oxycodone), risk_score=1.00, reasons: dispense-vs-waste quantity discrepancy, provider discrepancy rate is a statistical outlier vs. site peers, unusually high after-hours cabinet access frequency, missing required witness co-signature, elevated rate of cabinet overrides vs. peers, action: URGENT: escalate to pharmacy diversion review team within 24h
- **SITE_07 / SITE_07_RN020** -- event 9343 (Morphine), risk_score=1.00, reasons: dispense-vs-waste quantity discrepancy, provider discrepancy rate is a statistical outlier vs. site peers, unusually high after-hours cabinet access frequency, missing required witness co-signature, elevated rate of cabinet overrides vs. peers, action: URGENT: escalate to pharmacy diversion review team within 24h
- **SITE_01 / SITE_01_RN019** -- event 1128 (Hydromorphone), risk_score=1.00, reasons: dispense-vs-waste quantity discrepancy, provider discrepancy rate is a statistical outlier vs. site peers, unusually high after-hours cabinet access frequency, missing required witness co-signature, elevated rate of cabinet overrides vs. peers, action: URGENT: escalate to pharmacy diversion review team within 24h

See README.md for full discussion of the sensitivity/specificity tradeoff.
