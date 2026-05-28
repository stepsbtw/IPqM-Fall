#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("IPqM-Fall/data")
for f in sorted(data_dir.glob("*.csv")):
    df = pd.read_csv(f)
    if "Magnitude" not in df.columns:
        continue
    if f.name.endswith("_acceleration.csv"):
        df.rename(columns={"Magnitude": "amag"}, inplace=True)
    elif f.name.endswith("_angular_speed.csv"):
        df.rename(columns={"Magnitude": "wmag"}, inplace=True)
    else:
        continue
    df.to_csv(f, index=False)
    print("Updated", f)