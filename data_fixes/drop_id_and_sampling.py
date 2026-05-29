import pandas as pd
from pathlib import Path

base_path = Path("dataset/IPqM-Fall-parquet")

# Iterate through all ID folders
for id_folder in sorted(base_path.glob("ID*")):
    for positioning in ["CHEST", "LEFT", "RIGHT"]:

        # Drop 'id' column from sampling parquet files
        sampling_file = (
            id_folder
            / positioning
            / f"{id_folder.name}_{positioning}_sampling.parquet"
        )

        if sampling_file.exists():
            df = pd.read_parquet(sampling_file)

            if "id" in df.columns:
                df = df.drop(columns=["id"])
                df.to_parquet(sampling_file, index=False)
                print(f"Processed: {sampling_file}")

        # Drop 'sampling' column from acceleration parquet files
        accel_file = (
            id_folder
            / positioning
            / f"{id_folder.name}_{positioning}_acceleration.parquet"
        )

        if accel_file.exists():
            df = pd.read_parquet(accel_file)

            if "sampling" in df.columns:
                df = df.drop(columns=["sampling"])
                df.to_parquet(accel_file, index=False)
                print(f"Processed: {accel_file}")

        # Drop 'sampling' column from angular_speed parquet files
        angular_file = (
            id_folder
            / positioning
            / f"{id_folder.name}_{positioning}_angular_speed.parquet"
        )

        if angular_file.exists():
            df = pd.read_parquet(angular_file)

            if "sampling" in df.columns:
                df = df.drop(columns=["sampling"])
                df.to_parquet(angular_file, index=False)
                print(f"Processed: {angular_file}")