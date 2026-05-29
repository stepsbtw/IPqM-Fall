import pandas as pd
from pathlib import Path

PARQUET_ROOT = Path("dataset/IPqM-Fall-parquet")
OUT_ROOT = Path("dataset/IPqM-Fall-trials")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Find all signal parquet files recursively
for parquet_file in sorted(PARQUET_ROOT.rglob("*.parquet")):

    # Skip sampling files
    if parquet_file.name.endswith("_sampling.parquet"):
        continue

    stem = parquet_file.stem
    parts = stem.split("_")

    if len(parts) < 3:
        print(f"Invalid filename: {parquet_file}")
        continue

    person = parts[0]
    position = parts[1]
    signal = "_".join(parts[2:])

    # Matching sampling file
    sampling_file = (
        parquet_file.parent /
        f"{person}_{position}_sampling.parquet"
    )

    if not sampling_file.exists():
        print(f"Missing sampling file for {stem}")
        continue

    print(f"\nProcessing {stem}")

    try:
        signal_df = pd.read_parquet(parquet_file)
        labels_df = pd.read_parquet(sampling_file)

        # Validate timestamp column
        if "timestamp" not in signal_df.columns:
            print(f"Missing timestamp column in {parquet_file}")
            continue

        for _, row in labels_df.iterrows():

            exercise = row["exercise"]
            trial = row["trial"]

            begin_ts = row["beginning"]
            end_ts = row["ending"]

            segment = signal_df[
                (signal_df["timestamp"] >= begin_ts) &
                (signal_df["timestamp"] <= end_ts)
            ]

            if len(segment) == 0:
                continue

            # Safety cleanup
            segment = (
                segment
                .sort_values("timestamp")
                .drop_duplicates(subset="timestamp")
                .reset_index(drop=True)
            )

            out_name = (
                f"{person}_"
                f"{position}_"
                f"{exercise}_"
                f"trial{trial}_"
                f"{signal}.parquet"
            )

            segment.to_parquet(
                OUT_ROOT / out_name,
                index=False
            )

            print(
                f"Saved {out_name} "
                f"({len(segment)} samples)"
            )

    except Exception as e:
        print(f"SKIP {parquet_file}: {e}")