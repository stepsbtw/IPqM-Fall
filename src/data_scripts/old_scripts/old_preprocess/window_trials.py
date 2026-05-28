import json
import numpy as np
import pandas as pd

from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

TRIALS_ROOT = Path("IPqM-Fall/trials")
OUT_JSON = Path("labelstudio_windows.json")

TARGET_HZ = 90

WINDOW_SEC = 2
STRIDE_SEC = 1

WINDOW_SIZE = TARGET_HZ * WINDOW_SEC   # 180
STRIDE = TARGET_HZ * STRIDE_SEC        # 90

# =========================================================
# RESAMPLE TO FIXED 90 HZ
# =========================================================

def resample_to_90hz(df):

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )

    timestamps = df["timestamp"].values.astype(np.float64)

    if len(timestamps) < 2:
        return None

    start_ts = timestamps[0]
    end_ts = timestamps[-1]

    # 90 Hz -> 11.111... ms
    target_step = 1000 / TARGET_HZ

    new_timestamps = np.arange(
        start_ts,
        end_ts,
        target_step
    )

    if len(new_timestamps) < WINDOW_SIZE:
        return None

    numeric_cols = [
        c for c in df.columns
        if c != "timestamp"
    ]

    out = {
        "timestamp": new_timestamps
    }

    # Linear interpolation
    for col in numeric_cols:

        out[col] = np.interp(
            new_timestamps,
            timestamps,
            df[col].values
        )

    return pd.DataFrame(out)

# =========================================================
# CREATE LABEL STUDIO TASKS
# =========================================================

tasks = []

for parquet_file in sorted(TRIALS_ROOT.glob("*.parquet")):

    print(f"Processing {parquet_file.name}")

    try:

        df = pd.read_parquet(parquet_file)

        if len(df) < 10:
            continue

        # -------------------------------------------------
        # Resample
        # -------------------------------------------------

        df = resample_to_90hz(df)

        if df is None:
            continue

        # -------------------------------------------------
        # Sliding windows
        # -------------------------------------------------

        for start in range(
            0,
            len(df) - WINDOW_SIZE + 1,
            STRIDE
        ):

            end = start + WINDOW_SIZE

            window = df.iloc[start:end]

            values = {}

            for col in window.columns:

                if col == "timestamp":
                    continue

                values[col] = (
                    window[col]
                    .astype(float)
                    .round(6)
                    .tolist()
                )

            task = {
                "data": {
                    "source_file": parquet_file.name,
                    "window_start": int(start),
                    "window_end": int(end),
                    "values": values
                }
            }

            tasks.append(task)

    except Exception as e:

        print(f"ERROR: {parquet_file.name}")
        print(e)

print(f"\nGenerated {len(tasks)} windows")

# =========================================================
# SAVE LABEL STUDIO JSON
# =========================================================

with open(OUT_JSON, "w") as f:

    json.dump(tasks, f)

print(f"Saved {OUT_JSON}")