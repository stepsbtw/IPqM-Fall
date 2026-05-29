import pandas as pd
from pathlib import Path

base_path = Path("dataset/IPqM-Fall-parquet")

# Iterate through all ID folders
for id_folder in sorted(base_path.glob("ID*")):
    for positioning in ["CHEST", "LEFT", "RIGHT"]:

        sampling_file = (
            id_folder
            / positioning
            / f"{id_folder.name}_{positioning}_sampling.parquet"
        )

        if sampling_file.exists():
            try:
                df = pd.read_parquet(sampling_file)

                # Create trial column:
                # number repetitions of same exercise/userId/positioning/withRifle
                df["trial"] = (
                    df.groupby(
                        ["exercise", "userId", "positioning", "withRifle"]
                    ).cumcount() + 1
                )

                # Reorder columns to put trial after exercise
                cols = [
                    "exercise",
                    "trial",
                    "userId",
                    "positioning",
                    "withRifle",
                    "beginning",
                    "ending",
                ]

                # Keep only existing columns safely
                cols = [c for c in cols if c in df.columns]

                df = df[cols]

                df.to_parquet(sampling_file, index=False)

                print(f"Processed: {sampling_file}")

            except Exception as e:
                print(f"SKIP {sampling_file}: {e}")