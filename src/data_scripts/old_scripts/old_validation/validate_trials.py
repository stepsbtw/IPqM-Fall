import pandas as pd
from pathlib import Path
import numpy as np

TRIALS_ROOT = Path("IPqM-Fall/trials")

summary = []

for trial_file in sorted(TRIALS_ROOT.glob("*.parquet")):

    df = pd.read_parquet(trial_file)

    if len(df) == 0:
        print(f"\nEMPTY: {trial_file.name}")
        continue

    ts = df["timestamp"]

    # -------------------------------------------------
    # Basic stats
    # -------------------------------------------------

    n = len(df)

    start_ts = ts.iloc[0]
    end_ts = ts.iloc[-1]

    duration_sec = (end_ts - start_ts) / 1000

    # -------------------------------------------------
    # Timestamp checks
    # -------------------------------------------------

    diffs = ts.diff().dropna()

    monotonic = (diffs > 0).all()

    duplicate_ts = ts.duplicated().sum()

    max_gap_ms = diffs.max()

    median_gap_ms = diffs.median()

    # Estimated sampling rate
    if median_gap_ms > 0:
        estimated_hz = 1000 / median_gap_ms
    else:
        estimated_hz = np.nan

    # -------------------------------------------------
    # Flags
    # -------------------------------------------------

    issues = []

    if not monotonic:
        issues.append("non_monotonic")

    if duplicate_ts > 0:
        issues.append(f"duplicates={duplicate_ts}")

    if max_gap_ms > 200:
        issues.append(f"large_gap={max_gap_ms:.1f}ms")

    if duration_sec < 1:
        issues.append("too_short")

    if estimated_hz < 70:
        issues.append(f"low_hz={estimated_hz:.1f}")

    # -------------------------------------------------
    # Print suspicious trials
    # -------------------------------------------------

    if issues:

        print("\n========================")
        print(trial_file.name)

        print("Issues:", ", ".join(issues))

        print(f"Samples      : {n}")
        print(f"Duration (s) : {duration_sec:.2f}")
        print(f"Median dt ms : {median_gap_ms:.2f}")
        print(f"Max gap ms   : {max_gap_ms:.2f}")
        print(f"Estimated Hz : {estimated_hz:.2f}")

    # -------------------------------------------------
    # Store summary
    # -------------------------------------------------

    summary.append({
        "file": trial_file.name,
        "samples": n,
        "duration_sec": duration_sec,
        "median_gap_ms": median_gap_ms,
        "max_gap_ms": max_gap_ms,
        "estimated_hz": estimated_hz,
        "duplicates": duplicate_ts,
        "monotonic": monotonic,
        "issues": ";".join(issues)
    })

# =====================================================
# SAVE SUMMARY
# =====================================================

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    "trial_validation_summary.csv",
    index=False
)

print("\nSaved trial_validation_summary.csv")