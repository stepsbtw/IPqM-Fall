import sys
from pathlib import Path
import pandas as pd

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/IPqM-Fall")
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dataset/IPqM-Fall-parquet")

for f in sorted(src.rglob("*.csv")):
    try:
        # Read CSV
        df = pd.read_csv(f)

        # Preserve relative directory structure
        rel = f.relative_to(src)
        out = dst / rel.with_suffix(".parquet")

        # Create output directory
        out.parent.mkdir(parents=True, exist_ok=True)

        # Write parquet
        df.to_parquet(out, index=False)

        print("Wrote", out)

    except Exception as e:
        print("SKIP", f, ":", e)