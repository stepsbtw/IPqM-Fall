from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
DATASET_ROOT = Path("/home/caio-torkst/projects/tcc/IPqM-Fall/organized")
WINDOWS_CSV = "IPqM-Fall/windows_2_1.csv"
OUTPUT_DIR = Path("IPqM-Fall/fall_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = ["ax", "ay", "az", "amag", "wx", "wy", "wz", "wmag"]
FALL_CLASSES = {"FALL_1", "FALL_2", "FALL_3", "FALL_5", "FALL_6"}
SENSORS = ["CHEST", "LEFT", "RIGHT"]

# --- Load Windows Metadata ---
print("Reading metadata CSV...")
windows_df = pd.read_csv(WINDOWS_CSV)

# Cache parquet files to avoid redundant disk reads
parquet_cache = {}

for sensor in SENSORS:
    print(f"\n--- Processing Sensor: {sensor} ---")
    
    # Filter metadata for current sensor
    sensor_df = windows_df[windows_df["sensor_pos"] == sensor]
    
    if sensor_df.empty:
        print(f"No data found for sensor {sensor}. Skipping.")
        continue

    X_list = []
    y_list = []
    groups_list = []

    # Extract windows
    for _, row in tqdm(sensor_df.iterrows(), total=len(sensor_df), desc=f"Extracting {sensor} windows"):
        parquet_path = DATASET_ROOT / row["file"]

        if parquet_path not in parquet_cache:
            parquet_cache[parquet_path] = pd.read_parquet(parquet_path)

        df = parquet_cache[parquet_path]
        
        # Slice the window
        window = df.iloc[row["start_idx"]:row["end_idx"]]
        features = window[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        
        # Binary assignment: 1 for Fall, 0 for ADL
        label = 1 if row["label"] in FALL_CLASSES else 0

        X_list.append(features)
        y_list.append(label)
        groups_list.append(row["subject_id"])

    # Convert to standard NumPy arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(groups_list)

    # --- Save Prepared Data ---
    sensor_lower = sensor.lower()
    
    np.save(OUTPUT_DIR / f"X_{sensor_lower}.npy", X)
    np.save(OUTPUT_DIR / f"y_{sensor_lower}.npy", y)
    np.save(OUTPUT_DIR / f"groups_{sensor_lower}.npy", groups)
    
    print(f"Saved {sensor} datasets to {OUTPUT_DIR}/")
    print(f"X shape: {X.shape} | y shape: {y.shape} | groups shape: {groups.shape}")

print("\nAll datasets prepared and saved successfully!")