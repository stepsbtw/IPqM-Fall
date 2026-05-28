import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================

TRIALS_ROOT = Path("IPqM-Fall/OLD/trials")
OUT_ROOT = Path("IPqM-Fall/trials_90hz")

OUT_ROOT.mkdir(exist_ok=True)

TARGET_HZ = 90
TARGET_PERIOD_NS = int(1e9 / TARGET_HZ)

# ============================================
# RESAMPLING
# ============================================

trial_files = sorted(TRIALS_ROOT.glob("*.parquet"))

for parquet_file in tqdm(trial_files):

    df = pd.read_parquet(parquet_file)

    if len(df) < 2:
        continue

    # ----------------------------------------
    # Convert timestamp to datetime
    # ----------------------------------------

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="ms"
    )

    df = df.set_index("datetime")

    # ----------------------------------------
    # Build target timeline (90 Hz)
    # ----------------------------------------

    start = df.index.min()
    end = df.index.max()

    target_index = pd.date_range(
        start=start,
        end=end,
        freq=f"{TARGET_PERIOD_NS}ns"
    )

    # ----------------------------------------
    # Interpolate to target timeline
    # ----------------------------------------

    df_resampled = (
        df.reindex(df.index.union(target_index))
        .interpolate(method="time")
        .loc[target_index]
    )

    # ----------------------------------------
    # Restore timestamp column
    # ----------------------------------------

    df_resampled["timestamp"] = (
        df_resampled.index.astype("int64") // 10**6
    )

    df_resampled = df_resampled.reset_index(drop=True)

    # ----------------------------------------
    # Save
    # ----------------------------------------

    out_path = OUT_ROOT / parquet_file.name

    df_resampled.to_parquet(
        out_path,
        index=False
    )

print("Done.")