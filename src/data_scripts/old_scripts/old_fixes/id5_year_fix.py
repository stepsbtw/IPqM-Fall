import pandas as pd
from pathlib import Path

ROOT = Path("/home/caio-torkst/projects/tcc/IPqM-Fall")

# =========================================================
# CHECK ID5 CONSISTENCY AGAINST CHEST
# =========================================================

reference = pd.read_parquet(
    ROOT / "OLD/parquet/ID5_CHEST_acceleration.parquet"
)

ref_start = reference["timestamp"].min()
ref_end = reference["timestamp"].max()

print("\nREFERENCE (CHEST ACC)")
print(ref_start, ref_end)

targets = [
    "ID5_LEFT_acceleration",
    "ID5_LEFT_angular_speed",
    "ID5_RIGHT_acceleration",
    "ID5_RIGHT_angular_speed",
]

for name in targets:

    pq_path = ROOT / f"OLD/parquet/{name}.parquet"

    df = pd.read_parquet(pq_path)

    start = df["timestamp"].min()
    end = df["timestamp"].max()

    offset = ref_start - start

    shifted_start = start + offset
    shifted_end = end + offset

    print("\n====================")
    print(name)

    print("ORIGINAL")
    print(start, end)

    print("OFFSET DAYS")
    print(offset / 1000 / 60 / 60 / 24)

    print("SHIFTED")
    print(shifted_start, shifted_end)

    print("MATCHES CHEST?")
    print(
        abs(shifted_start - ref_start) < 1000 and
        abs(shifted_end - ref_end) < 1000
    )