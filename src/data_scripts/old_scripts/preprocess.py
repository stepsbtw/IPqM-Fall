#!/usr/bin/env python3
"""Orchestrate full preprocessing pipeline.

This script runs the repository preprocessing steps in the documented
order (fixes → csv->parquet → flatten/segment → downsample → windows).

It executes the existing scripts as subprocesses to avoid changing their
implementation details.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STEP_COMMANDS = {
    "fix": lambda root, dry_run: [
        sys.executable,
        "scripts/fix_dataset.py",
        "--root",
        str(root),
    ] + (["--dry-run"] if dry_run else []),

    "csv2parquet": lambda root, dry_run: [
        sys.executable,
        "scripts/csv_to_parquet.py",
    ],

    "flatten": lambda root, dry_run: [
        sys.executable,
        "scripts/flatten.py",
    ],

    "segment": lambda root, dry_run: [
        sys.executable,
        "scripts/segment_trials.py",
    ],

    "downsample": lambda root, dry_run: [
        sys.executable,
        "downsample_trials.py",
    ],

    "windows": lambda root, dry_run: [
        sys.executable,
        "generate_windows.py",
    ],

    "labelstudio": lambda root, dry_run: [
        sys.executable,
        "scripts/window_trials.py",
    ],

    "app": lambda root, dry_run: [
        "streamlit",
        "run",
        "app.py",
    ],
}


def run_cmd(cmd: list[str], continue_on_error: bool) -> int:
    print("\n>> Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
        if not continue_on_error:
            raise SystemExit(proc.returncode)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full preprocessing pipeline.")

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("IPqM-Fall"),
        help="Dataset root directory (default: IPqM-Fall)",
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(STEP_COMMANDS.keys()),
        default=["fix", "csv2parquet", "flatten", "segment", "downsample", "windows"],
        help="Steps to run (default: full pipeline)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass dry-run to the fixer (no parquet writes for id5-offset)",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue executing remaining steps even if one fails",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = args.root

    print("Preprocess pipeline starting. Dataset root:", root)

    for step in args.steps:
        if step not in STEP_COMMANDS:
            print("Unknown step:", step)
            raise SystemExit(2)

        cmd = STEP_COMMANDS[step](root, args.dry_run)

        try:
            run_cmd(cmd, continue_on_error=args.continue_on_error)
        except SystemExit as e:
            if args.continue_on_error:
                print(f"Step {step} failed (exit {e.code}); continuing")
                continue
            raise

    print("\nPreprocess pipeline finished.")


if __name__ == "__main__":
    main()
