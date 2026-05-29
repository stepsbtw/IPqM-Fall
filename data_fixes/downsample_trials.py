import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================

TRIALS_ROOT = Path("dataset/IPqM-Fall-trials")
OUT_ROOT = Path("dataset/IPqM-Fall-trials-90hz")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_HZ = 90
TARGET_PERIOD_NS = int(1e9 / TARGET_HZ)

# ============================================
# RESAMPLING
# ============================================

trial_files = sorted(TRIALS_ROOT.rglob("*.parquet"))

for parquet_file in tqdm(trial_files):

    try:
        df = pd.read_parquet(parquet_file)

        if len(df) < 2:
            continue

        # ----------------------------------------
        # Validate timestamp column
        # ----------------------------------------

        if "timestamp" not in df.columns:
            print(f"SKIP no timestamp: {parquet_file}")
            continue

        # ----------------------------------------
        # Convert timestamp to datetime
        # ----------------------------------------

        df["datetime"] = pd.to_datetime(
            df["timestamp"],
            unit="ms"
        )

        df = (
            df
            .sort_values("datetime")
            .drop_duplicates(subset="datetime")
            .set_index("datetime")
        )

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

        # Remove helper column if present
        if "datetime" in df_resampled.columns:
            df_resampled = df_resampled.drop(columns=["datetime"])

        # ----------------------------------------
        # Save
        # ----------------------------------------

        out_path = OUT_ROOT / parquet_file.name

        df_resampled.to_parquet(
            out_path,
            index=False
        )

    except Exception as e:
        print(f"SKIP {parquet_file}: {e}")

print("Done.")