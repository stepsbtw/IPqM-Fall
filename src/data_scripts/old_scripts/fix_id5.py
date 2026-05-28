import pandas as pd
from pathlib import Path

OFFSET_MS = 1711022512376 - 1648474368666

sampling_files = [
    "IPqM-Fall/OLD/sampling/ID5_LEFT_sampling.csv",
    "IPqM-Fall/OLD/sampling/ID5_RIGHT_sampling.csv"
]

for file in sampling_files:

    df = pd.read_csv(file)

    df["beginning"] += OFFSET_MS
    df["ending"] += OFFSET_MS

    df.to_csv(file, index=False)

    print(f"Fixed {file}")

    print(
        df["beginning"].min(),
        df["ending"].max()
    )