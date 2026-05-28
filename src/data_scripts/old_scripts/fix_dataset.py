#!/usr/bin/env python3
"""Unified dataset fixes.

This script consolidates dataset-fix routines previously spread across
multiple scripts into a single command-line entrypoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


POSITIONS = ("CHEST", "LEFT", "RIGHT")


def safe_drop_column(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, bool]:
    """Drop column if it exists. Returns (df, changed)."""
    if column not in df.columns:
        return df, False
    return df.drop(columns=[column]), True


def fix_raw_columns(raw_root: Path) -> int:
    """Drop stale columns from raw sampling/acc/gyro files."""
    changed_files = 0

    for id_folder in sorted(raw_root.glob("ID*")):
        for position in POSITIONS:
            sampling_file = id_folder / position / f"{id_folder.name}_{position}_sampling.csv"
            if sampling_file.exists():
                df = pd.read_csv(sampling_file)
                df, changed = safe_drop_column(df, "id")
                if changed:
                    df.to_csv(sampling_file, index=False)
                    changed_files += 1
                    print(f"Updated: {sampling_file}")

            accel_file = id_folder / position / f"{id_folder.name}_{position}_acceleration.csv"
            if accel_file.exists():
                df = pd.read_csv(accel_file)
                df, changed = safe_drop_column(df, "sampling")
                if changed:
                    df.to_csv(accel_file, index=False)
                    changed_files += 1
                    print(f"Updated: {accel_file}")

            angular_file = id_folder / position / f"{id_folder.name}_{position}_angular_speed.csv"
            if angular_file.exists():
                df = pd.read_csv(angular_file)
                df, changed = safe_drop_column(df, "sampling")
                if changed:
                    df.to_csv(angular_file, index=False)
                    changed_files += 1
                    print(f"Updated: {angular_file}")

    return changed_files


def add_trial_to_sampling(raw_root: Path) -> int:
    """Add and recompute trial column in raw sampling files."""
    changed_files = 0

    for id_folder in sorted(raw_root.glob("ID*")):
        for position in POSITIONS:
            sampling_file = id_folder / position / f"{id_folder.name}_{position}_sampling.csv"
            if not sampling_file.exists():
                continue

            df = pd.read_csv(sampling_file)

            required = ["exercise", "userId", "positioning", "withRifle"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"Skipped {sampling_file}: missing columns {missing}")
                continue

            df["trial"] = df.groupby(required).cumcount() + 1

            desired = [
                "exercise",
                "trial",
                "userId",
                "positioning",
                "withRifle",
                "beginning",
                "ending",
            ]
            existing = [c for c in desired if c in df.columns]
            extras = [c for c in df.columns if c not in existing]
            df = df[existing + extras]

            df.to_csv(sampling_file, index=False)
            changed_files += 1
            print(f"Updated: {sampling_file}")

    return changed_files


def rename_magnitude_columns(data_root: Path) -> int:
    """Rename Magnitude to amag/wmag depending on signal file type."""
    changed_files = 0

    for csv_file in sorted(data_root.glob("*.csv")):
        df = pd.read_csv(csv_file)
        if "Magnitude" not in df.columns:
            continue

        if csv_file.name.endswith("_acceleration.csv"):
            df = df.rename(columns={"Magnitude": "amag"})
        elif csv_file.name.endswith("_angular_speed.csv"):
            df = df.rename(columns={"Magnitude": "wmag"})
        else:
            continue

        df.to_csv(csv_file, index=False)
        changed_files += 1
        print(f"Updated: {csv_file}")

    return changed_files


def sort_sampling_files(sampling_root: Path) -> int:
    """Sort sampling files by beginning timestamp."""
    changed_files = 0

    for file_path in sorted(sampling_root.glob("*_sampling.csv")):
        df = pd.read_csv(file_path)
        if "beginning" not in df.columns:
            print(f"Skipped {file_path}: missing 'beginning' column")
            continue

        sorted_df = df.sort_values("beginning").reset_index(drop=True)
        if sorted_df.equals(df.reset_index(drop=True)):
            continue

        sorted_df.to_csv(file_path, index=False)
        changed_files += 1
        print(f"Updated: {file_path}")

    return changed_files


def fix_id5_timestamp_offset(parquet_root: Path, dry_run: bool) -> int:
    """Align ID5 LEFT/RIGHT parquet timestamps to ID5 CHEST reference."""
    changed_files = 0

    reference_path = parquet_root / "ID5_CHEST_acceleration.parquet"
    if not reference_path.exists():
        print(f"Skipped ID5 fix: missing reference {reference_path}")
        return 0

    reference = pd.read_parquet(reference_path)
    ref_start = reference["timestamp"].min()
    ref_end = reference["timestamp"].max()

    targets = [
        "ID5_LEFT_acceleration",
        "ID5_LEFT_angular_speed",
        "ID5_RIGHT_acceleration",
        "ID5_RIGHT_angular_speed",
    ]

    print("ID5 reference (CHEST):", ref_start, ref_end)

    for name in targets:
        pq_path = parquet_root / f"{name}.parquet"
        if not pq_path.exists():
            print(f"Skipped {name}: file not found")
            continue

        df = pd.read_parquet(pq_path)
        if "timestamp" not in df.columns:
            print(f"Skipped {name}: missing 'timestamp' column")
            continue

        start = df["timestamp"].min()
        end = df["timestamp"].max()
        offset = ref_start - start

        shifted_start = start + offset
        shifted_end = end + offset

        matches = abs(shifted_start - ref_start) < 1000 and abs(shifted_end - ref_end) < 1000
        print(f"{name}: offset_days={offset / 1000 / 60 / 60 / 24:.4f}, aligns={matches}")

        if offset == 0:
            continue

        if dry_run:
            changed_files += 1
            continue

        df = df.copy()
        df["timestamp"] = df["timestamp"] + offset
        df.to_parquet(pq_path, index=False)
        changed_files += 1
        print(f"Updated: {pq_path}")

    return changed_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset fixes from one script.")

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("IPqM-Fall"),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["raw-columns", "trial", "rename-mag", "sort-sampling", "id5-offset"],
        default=["raw-columns", "trial", "rename-mag", "sort-sampling", "id5-offset"],
        help="Fix steps to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without writing files for the id5-offset step.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = args.root
    raw_root = root / "raw"
    data_root = root / "data"
    sampling_root = root / "sampling"
    parquet_root = root / "OLD" / "parquet"

    total = 0

    if "raw-columns" in args.steps:
        changed = fix_raw_columns(raw_root)
        print(f"raw-columns: {changed} files updated")
        total += changed

    if "trial" in args.steps:
        changed = add_trial_to_sampling(raw_root)
        print(f"trial: {changed} files updated")
        total += changed

    if "rename-mag" in args.steps:
        changed = rename_magnitude_columns(data_root)
        print(f"rename-mag: {changed} files updated")
        total += changed

    if "sort-sampling" in args.steps:
        changed = sort_sampling_files(sampling_root)
        print(f"sort-sampling: {changed} files updated")
        total += changed

    if "id5-offset" in args.steps:
        changed = fix_id5_timestamp_offset(parquet_root, dry_run=args.dry_run)
        suffix = " (dry-run)" if args.dry_run else ""
        print(f"id5-offset{suffix}: {changed} files affected")
        total += changed

    print(f"Total files affected: {total}")


if __name__ == "__main__":
    main()