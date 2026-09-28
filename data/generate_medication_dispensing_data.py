"""
Synthetic medication dispensing/administration event generator.

Simulates controlled-substance dispensing cabinet (Pyxis-like) events across
8 hospital sites, for a fixed pool of nurses/providers per site. Produces a
realistic base rate of normal dispensing behavior, PLUS a labeled minority
subset of injected "diversion" patterns:

  - dispense-vs-waste quantity discrepancies (dispensed - administered - wasted > 0)
  - unusually high after-hours access frequency concentrated on one provider
  - repeated cabinet overrides without a required co-signature/witness
  - a provider's overall dispensing rate statistically anomalous vs. site peers

ALL DATA IS SYNTHETIC. No real patients, providers, or hospital records are
used or represented here.

Ground truth: every row carries `is_diversion` (1/0), the actual generative
label used to inject the anomalous pattern. This is the label the detector
in `surveillance/detection_rules.py` is evaluated against — it is NOT
computed from the detection rules themselves.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42
SITES = [f"SITE_{i+1:02d}" for i in range(8)]
PROVIDERS_PER_SITE = 25
DIVERTING_PROVIDERS_PER_SITE = 2  # genuinely diverting providers, injected
EVENTS_PER_PROVIDER_MEAN = 55  # -> thousands of events total across 8 sites

MEDICATIONS = [
    ("Fentanyl", 0.1, "mcg"),
    ("Oxycodone", 10, "mg"),
    ("Hydromorphone", 2, "mg"),
    ("Morphine", 4, "mg"),
    ("Midazolam", 2, "mg"),
    ("Lorazepam", 1, "mg"),
]


def _hour_is_after_hours(hour):
    return (hour >= 23) or (hour < 6)


def generate_dataset(seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    event_id = 0

    for site in SITES:
        provider_ids = [f"{site}_RN{p:03d}" for p in range(PROVIDERS_PER_SITE)]
        diverting_providers = set(
            rng.choice(provider_ids, size=DIVERTING_PROVIDERS_PER_SITE, replace=False)
        )

        for provider in provider_ids:
            is_diverting_provider = provider in diverting_providers
            n_events = max(5, int(rng.normal(EVENTS_PER_PROVIDER_MEAN, 12)))

            for _ in range(n_events):
                event_id += 1
                med_name, unit_dose, unit = MEDICATIONS[rng.integers(0, len(MEDICATIONS))]

                # timestamp: random day over a 90-day surveillance window
                day_offset = int(rng.integers(0, 90))
                if is_diverting_provider and rng.random() < 0.35:
                    # diverting providers show elevated after-hours access
                    hour = int(rng.choice([23, 0, 1, 2, 3, 4, 5]))
                else:
                    # normal shift-weighted hours (day/evening shifts dominate)
                    hour = int(rng.choice(range(24), p=_shift_hour_probs()))
                minute = int(rng.integers(0, 60))

                dispensed_qty = round(float(unit_dose * rng.uniform(1, 4)), 2)

                # baseline: small/no discrepancy, co-signed most of the time
                discrepancy = 0.0
                witnessed = rng.random() < 0.94
                override = rng.random() < 0.05

                event_is_diversion = 0

                if is_diverting_provider:
                    # each event from a diverting provider has an elevated
                    # (but not 100%) chance of carrying an actual diversion
                    # pattern -- diversion is intermittent/hidden, not constant
                    if rng.random() < 0.55:
                        pattern = rng.choice(
                            ["waste_discrepancy", "no_cosign_override", "combo"]
                        )
                        if pattern in ("waste_discrepancy", "combo"):
                            discrepancy = round(
                                dispensed_qty * rng.uniform(0.25, 0.9), 2
                            )
                        if pattern in ("no_cosign_override", "combo"):
                            override = True
                            witnessed = rng.random() < 0.10  # override w/o co-sign
                        event_is_diversion = 1

                administered_qty = round(dispensed_qty - discrepancy, 2)
                wasted_qty = 0.0 if discrepancy == 0 else round(
                    discrepancy * rng.uniform(0.0, 0.3), 2
                )  # partially "wasted-on-paper" to obscure the gap
                unaccounted_qty = round(dispensed_qty - administered_qty - wasted_qty, 2)

                rows.append(
                    {
                        "event_id": event_id,
                        "site": site,
                        "provider_id": provider,
                        "day_offset": day_offset,
                        "hour": hour,
                        "minute": minute,
                        "after_hours": int(_hour_is_after_hours(hour)),
                        "medication": med_name,
                        "unit": unit,
                        "dispensed_qty": dispensed_qty,
                        "administered_qty": administered_qty,
                        "wasted_qty": wasted_qty,
                        "unaccounted_qty": unaccounted_qty,
                        "witnessed_cosign": int(witnessed),
                        "cabinet_override": int(override),
                        "is_diversion": event_is_diversion,
                        "diverting_provider_ground_truth": int(is_diverting_provider),
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def _shift_hour_probs():
    # weight day/evening hours (7am-11pm) much higher than overnight
    probs = np.ones(24)
    for h in range(24):
        if 7 <= h <= 22:
            probs[h] = 3.0
        else:
            probs[h] = 0.3
    return probs / probs.sum()


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/dispensing_events.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} events across {df['site'].nunique()} sites")
    print(f"Diversion-labeled events: {df['is_diversion'].sum():,} "
          f"({df['is_diversion'].mean()*100:.2f}% of all events)")
    print(f"Diverting providers (ground truth): "
          f"{df.groupby('site')['diverting_provider_ground_truth'].apply(lambda s: s.groupby(df.loc[s.index,'provider_id']).max().sum())}")
    print(f"Wrote {out_path}")
