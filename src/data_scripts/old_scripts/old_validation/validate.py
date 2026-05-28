import pandas as pd
from pathlib import Path

PARQUET_ROOT = Path("IPqM-Fall/OLD/parquet")

groups = {}

# Group files by ID + position
for parquet_file in PARQUET_ROOT.glob("*.parquet"):

    parts = parquet_file.stem.split("_")

    person = parts[0]
    position = parts[1]
    signal = "_".join(parts[2:])

    key = (person, position)

    groups.setdefault(key, {})
    groups[key][signal] = parquet_file

# Compare accel vs gyro
for key, files in groups.items():

    if (
        "acceleration" not in files or
        "angular_speed" not in files
    ):
        continue

    acc = pd.read_parquet(files["acceleration"])
    gyro = pd.read_parquet(files["angular_speed"])

    print(f"\n{key}")

    print("ACC")
    print(
        len(acc),
        acc["timestamp"].min(),
        acc["timestamp"].max()
    )

    print("GYRO")
    print(
        len(gyro),
        gyro["timestamp"].min(),
        gyro["timestamp"].max()
    )

    overlap_start = max(
        acc["timestamp"].min(),
        gyro["timestamp"].min()
    )

    overlap_end = min(
        acc["timestamp"].max(),
        gyro["timestamp"].max()
    )

    print(
        "Overlap duration (s):",
        (overlap_end - overlap_start) / 1000
    )