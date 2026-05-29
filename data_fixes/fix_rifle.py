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

        if not sampling_file.exists():
            continue

        try:
            df = pd.read_parquet(sampling_file)

            # Convert:
            # ADL_1 + withRifle=1 -> ADL1R
            # ADL_1 + withRifle=0 -> ADL1
            def normalize_exercise(row):
                ex = str(row["exercise"]).replace("_", "")

                if "withRifle" in row and row["withRifle"] == 1:
                    ex += "R"

                return ex

            df["exercise"] = df.apply(normalize_exercise, axis=1)

            # Recompute trial WITHOUT withRifle
            df["trial"] = (
                df.groupby(
                    ["exercise", "userId", "positioning"]
                ).cumcount() + 1
            )

            # Drop withRifle column
            if "withRifle" in df.columns:
                df = df.drop(columns=["withRifle"])

            # Reorder columns
            cols = [
                "exercise",
                "trial",
                "userId",
                "positioning",
                "beginning",
                "ending",
            ]

            cols = [c for c in cols if c in df.columns]

            df = df[cols]

            # Save back
            df.to_parquet(sampling_file, index=False)

            print(f"Processed: {sampling_file}")

        except Exception as e:
            print(f"SKIP {sampling_file}: {e}")