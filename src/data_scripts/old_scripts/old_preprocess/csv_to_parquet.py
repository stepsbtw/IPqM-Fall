import sys
from pathlib import Path
import pandas as pd

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("IPqM-Fall/OLD/csv")
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src
dst.mkdir(parents=True, exist_ok=True)

for f in sorted(src.glob("*.csv")):
    try:
        df = pd.read_csv(f)
        out = dst / (f.stem + ".parquet")
        df.to_parquet(out, index=False)
        print("Wrote", out)
    except Exception as e:
        print("SKIP", f, ":", e)