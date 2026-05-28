import pandas as pd
import os
from pathlib import Path

# Directory containing sampling files
sampling_dir = "/home/caio-torkst/projects/tcc/IPqM-Fall/sampling"

# Get all sampling CSV files
sampling_files = sorted(Path(sampling_dir).glob("*_sampling.csv"))

# Sort each file by beginning timestamp
for file_path in sampling_files:
    print(f"Sorting {file_path.name}...")
    
    # Read the CSV
    df = pd.read_csv(file_path)
    
    # Sort by beginning timestamp
    df = df.sort_values('beginning').reset_index(drop=True)
    
    # Write back to the same file
    df.to_csv(file_path, index=False)
    
    print(f"  ✓ Sorted {len(df)} rows")

print(f"\n✓ All {len(sampling_files)} sampling files have been sorted by beginning timestamp!")
