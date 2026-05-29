import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

TRIALS_ROOT = Path("dataset/IPqM-Fall-trials-90hz")
OUTPUT_ROOT = Path("dataset/IPqM-Fall-trials-90hz-combined")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# PROCESSING
# ============================================================

def process_combined_parquets():

    # Find all acceleration parquet files recursively
    acc_files = sorted(
        TRIALS_ROOT.rglob("*_acceleration.parquet")
    )

    if not acc_files:
        print("No acceleration parquet files found.")
        return

    print(f"Starting merge of {len(acc_files)} files...\n")

    successes = 0
    errors = 0

    for acc_path in tqdm(acc_files, desc="Processing"):

        # Matching gyro file
        gyro_name = acc_path.name.replace(
            "_acceleration",
            "_angular_speed"
        )

        gyro_path = acc_path.parent / gyro_name

        if not gyro_path.exists():
            print(f"Missing gyro pair for {acc_path.name}")
            errors += 1
            continue

        try:

            # ------------------------------------------------
            # Load parquet files
            # ------------------------------------------------

            df_acc = (
                pd.read_parquet(acc_path)
                .sort_values("timestamp")
                .drop_duplicates(subset="timestamp")
            )

            df_gyro = (
                pd.read_parquet(gyro_path)
                .sort_values("timestamp")
                .drop_duplicates(subset="timestamp")
            )

            # ------------------------------------------------
            # Validate timestamp column
            # ------------------------------------------------

            if (
                "timestamp" not in df_acc.columns or
                "timestamp" not in df_gyro.columns
            ):
                print(f"Missing timestamp column: {acc_path.name}")
                errors += 1
                continue

            # ------------------------------------------------
            # Temporal merge
            # ------------------------------------------------

            df_combined = pd.merge_asof(
                left=df_acc,
                right=df_gyro,
                on="timestamp",
                direction="nearest",
                tolerance=25
            )

            # ------------------------------------------------
            # Interpolate missing values
            # ------------------------------------------------

            df_combined = df_combined.interpolate(
                method="linear",
                limit_direction="both"
            )

            df_combined = df_combined.dropna()

            if len(df_combined) == 0:
                print(f"Empty after merge: {acc_path.name}")
                errors += 1
                continue

            # ------------------------------------------------
            # Save combined parquet
            # ------------------------------------------------

            combined_name = acc_path.name.replace(
                "_acceleration",
                ""
            )

            out_path = OUTPUT_ROOT / combined_name

            df_combined.to_parquet(
                out_path,
                index=False
            )

            successes += 1

        except Exception as e:

            print(f"SKIP {acc_path.name}: {e}")
            errors += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 50)
    print("PROCESS COMPLETED")
    print("=" * 50)
    print(f"Successful files: {successes}")

    if errors > 0:
        print(f"Skipped/errors: {errors}")

    print(f"\nOutput folder: {OUTPUT_ROOT}")


if __name__ == "__main__":
    process_combined_parquets()