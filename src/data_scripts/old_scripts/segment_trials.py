import pandas as pd
from pathlib import Path

PARQUET_ROOT = Path("IPqM-Fall/OLD/parquet")
SAMPLING_ROOT = Path("IPqM-Fall/OLD/sampling")
OUT_ROOT = Path("IPqM-Fall/trials")

OUT_ROOT.mkdir(exist_ok=True)

for parquet_file in PARQUET_ROOT.glob("*.parquet"):

    stem = parquet_file.stem
    parts = stem.split("_")

    person = parts[0]
    position = parts[1]
    signal = "_".join(parts[2:])

    sampling_file = (
        SAMPLING_ROOT /
        f"{person}_{position}_sampling.csv"
    )

    if not sampling_file.exists():
        print(f"Missing sampling file for {stem}")
        continue

    print(f"\nProcessing {stem}")

    signal_df = pd.read_parquet(parquet_file)
    labels_df = pd.read_csv(sampling_file)

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