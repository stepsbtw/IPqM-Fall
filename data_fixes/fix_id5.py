import pandas as pd
from pathlib import Path

OFFSET_MS = 1711022512376 - 1648474368666

base_path = Path("dataset/IPqM-Fall-parquet")

files = [
    base_path / "ID5" / "LEFT" / "ID5_LEFT_sampling.parquet",
    base_path / "ID5" / "RIGHT" / "ID5_RIGHT_sampling.parquet",
]

for file in files:

    if not file.exists():
        print(f"SKIP missing file: {file}")
        continue

    df = pd.read_parquet(file)

    df["beginning"] += OFFSET_MS
    df["ending"] += OFFSET_MS

    df.to_parquet(file, index=False)

    print(f"Fixed {file}")

    print(
        "beginning min:", df["beginning"].min(),
        "ending max:", df["ending"].max()
    )