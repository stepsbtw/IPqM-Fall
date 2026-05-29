import sys
from pathlib import Path
import pandas as pd

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/IPqM-Fall-parquet")

for f in sorted(data_dir.rglob("*.parquet")):
    try:
        df = pd.read_parquet(f)

        if "Magnitude" not in df.columns:
            continue

        if f.name.endswith("_acceleration.parquet"):
            df.rename(columns={"Magnitude": "amag"}, inplace=True)

        elif f.name.endswith("_angular_speed.parquet"):
            df.rename(columns={"Magnitude": "wmag"}, inplace=True)

        else:
            continue

        df.to_parquet(f, index=False)

        print("Updated", f)

    except Exception as e:
        print("SKIP", f, ":", e)