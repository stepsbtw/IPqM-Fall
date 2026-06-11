# src/generate_dataset.py

import os
import subprocess
import sys
from pathlib import Path


PIPELINE_STEPS = [
    "src/dataset/1_generate_windows.py",
    "src/dataset/2_fix_fall_labels.py",
    "src/dataset/3_fix_transitions.py",
    "src/dataset/4_fix_sitting.py",
    "src/dataset/5_generate_arrays.py",
]


def run_step(script_path: Path, project_root: Path):
    print("\n" + "=" * 80)
    print(f"Running: {script_path.name}")
    print("=" * 80)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=project_root,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )

    returncode = process.wait()

    if returncode != 0:
        raise RuntimeError(
            f"Pipeline failed at: {script_path}"
        )


def main():
    project_root = Path(__file__).resolve().parents[1]

    for step in PIPELINE_STEPS:
        script_path = project_root / step

        if not script_path.exists():
            raise FileNotFoundError(f"Missing script: {script_path}")

        run_step(script_path, project_root)

    print("\nDataset pipeline completed successfully.")


if __name__ == "__main__":
    main()