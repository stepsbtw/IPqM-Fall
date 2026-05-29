import pandas as pd
from pathlib import Path

# Root dataset directory
base_dir = Path("dataset/IPqM-Fall-parquet")

# Find all sampling parquet files recursively
sampling_files = sorted(base_dir.rglob("*_sampling.parquet"))

# Sort each file by beginning timestamp
for file_path in sampling_files:
    print(f"Sorting {file_path}...")

    try:
        # Read parquet
        df = pd.read_parquet(file_path)

        # Skip if column doesn't exist
        if "beginning" not in df.columns:
            print("  SKIP: no 'beginning' column")
            continue

        # Sort rows
        df = df.sort_values("beginning").reset_index(drop=True)

        # Save back
        df.to_parquet(file_path, index=False)

        print(f"  ✓ Sorted {len(df)} rows")

    except Exception as e:
        print(f"  SKIP: {e}")

print(f"\n✓ Processed {len(sampling_files)} sampling parquet files!")