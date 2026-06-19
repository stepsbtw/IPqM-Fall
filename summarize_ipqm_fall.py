from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TASK_NAMES = {
    "y_detect_fall": {0: "Fall", 1: "Non-Fall"},
    "y_classify_fall": {0: "Backward", 1: "Frontal", 2: "Lateral-left", 3: "Lateral-right"},
    "y_classify_posture": {0: "Standing", 1: "Sitting", 2: "Kneeling", 3: "Down/Prone"},
    "y_classify_movement": {0: "Walking", 1: "Sweeping", 2: "Running", 3: "Jumping", 4: "Crawling"},
    "y_classify_transition": {0: "To-kneeling", 1: "To-prone", 2: "Sit/stand"},
    "y_unified": {
        0: "Backward fall", 1: "Frontal fall", 2: "Lateral-left fall", 3: "Lateral-right fall",
        4: "Standing", 5: "Sitting", 6: "Kneeling", 7: "Down/Prone",
        8: "Walking", 9: "Sweeping", 10: "Running", 11: "Jumping", 12: "Crawling",
    },
}

WINDOW_RE = re.compile(r"(?P<window>\d+(?:\.\d+)?)-sec_(?P<stride>\d+(?:\.\d+)?)-step")
ACTIVITY_RE = re.compile(r"^ID\d+_(?P<activity>.+?)_TRIAL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize learning-ready IPqM-Fall arrays.")
    parser.add_argument("--root", type=Path, default=Path("IPqM-Fall/windowed"),
                        help="Directory containing folders such as 2-sec_1-step and 5-sec_1-step.")
    parser.add_argument("--output", type=Path, default=Path("dataset_summary"),
                        help="Directory where CSV and JSON summaries are written.")
    return parser.parse_args()


def load_array(path: Path) -> np.ndarray | None:
    return np.load(path, allow_pickle=False) if path.exists() else None


def class_name(task: str, class_id: int) -> str:
    return TASK_NAMES.get(task, {}).get(class_id, str(class_id))


def condition_from_sync_id(sync_id: str) -> str:
    match = ACTIVITY_RE.match(str(sync_id))
    if not match:
        return "Unknown"
    return "Armed" if match.group("activity").endswith("R") else "Unarmed"


def metadata_checks(dataset_root: Path, window: str, stride: str) -> dict:
    window_tag = window.replace(".0", "")
    stride_tag = stride.replace(".0", "")
    path = dataset_root / f"windows_{window_tag}_{stride_tag}.parquet"
    result = {"metadata_file": str(path), "metadata_available": path.exists()}
    if not path.exists():
        return result

    try:
        metadata = pd.read_parquet(path)
    except Exception as error:
        result["metadata_error"] = str(error)
        return result

    trial_num = metadata["file"].map(lambda value: Path(value).stem.rsplit("_TRIAL", 1)[1])
    metadata["sync_id"] = (
        metadata["subject_id"].astype(str) + "_" + metadata["activity_code"].astype(str)
        + "_TRIAL" + trial_num + "_win_" + metadata["start_idx"].astype(str)
    )
    group_sizes = metadata.groupby("sync_id").size()
    valid_ids = group_sizes[group_sizes == 3].index
    valid = metadata[metadata["sync_id"].isin(valid_ids)]
    disagreements = valid.groupby("sync_id")["label"].nunique().gt(1)

    result.update({
        "metadata_rows": int(len(metadata)),
        "synchronized_groups": int(len(group_sizes)),
        "complete_three_position_groups": int(len(valid_ids)),
        "incomplete_groups": int((group_sizes != 3).sum()),
        "complete_groups_with_label_disagreement": int(disagreements.sum()),
    })
    return result


def summarize_folder(folder: Path) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    match = WINDOW_RE.fullmatch(folder.name)
    if not match:
        raise ValueError(f"Invalid window folder name: {folder.name}")

    window = match.group("window")
    stride = match.group("stride")
    groups = load_array(folder / "groups.npy")
    sync_ids = load_array(folder / "sync_ids.npy")
    targets = {path.stem: load_array(path) for path in sorted(folder.glob("y_*.npy"))}

    if groups is None or sync_ids is None:
        raise FileNotFoundError(f"{folder} must contain groups.npy and sync_ids.npy")

    n = len(groups)
    if len(sync_ids) != n:
        raise ValueError(f"{folder}: groups.npy and sync_ids.npy have different lengths")
    for task, values in targets.items():
        if len(values) != n:
            raise ValueError(f"{folder}: {task}.npy has {len(values)} rows; expected {n}")

    conditions = np.array([condition_from_sync_id(value) for value in sync_ids])
    class_rows, task_rows, subject_rows, condition_rows = [], [], [], []

    for task, values in targets.items():
        valid = values >= 0
        task_rows.append({
            "window_seconds": float(window), "stride_seconds": float(stride), "task": task,
            "total_windows": n, "valid_windows": int(valid.sum()), "ignored_windows": int((~valid).sum()),
        })
        for class_id, count in zip(*np.unique(values[valid], return_counts=True)) if valid.any() else ([], []):
            class_rows.append({
                "window_seconds": float(window), "stride_seconds": float(stride), "task": task,
                "class_id": int(class_id), "class_name": class_name(task, int(class_id)),
                "count": int(count), "percentage_of_valid": 100.0 * int(count) / int(valid.sum()),
            })

    for subject in sorted(np.unique(groups), key=lambda value: int(re.search(r"\d+", str(value)).group())):
        mask = groups == subject
        row = {"window_seconds": float(window), "stride_seconds": float(stride),
               "subject": str(subject), "total_windows": int(mask.sum())}
        for task, values in targets.items():
            row[f"{task}_valid"] = int(np.sum(mask & (values >= 0)))
            row[f"{task}_ignored"] = int(np.sum(mask & (values < 0)))
        subject_rows.append(row)

    for condition in sorted(np.unique(conditions)):
        mask = conditions == condition
        row = {"window_seconds": float(window), "stride_seconds": float(stride),
               "condition": str(condition), "total_windows": int(mask.sum())}
        for task, values in targets.items():
            row[f"{task}_valid"] = int(np.sum(mask & (values >= 0)))
        condition_rows.append(row)

    checks = {
        "folder": str(folder), "window_seconds": float(window), "stride_seconds": float(stride),
        "total_learning_ready_windows": n, "subjects": sorted(map(str, np.unique(groups))),
        "target_files": sorted(targets),
    }

    if "y_detect_fall" in targets and "y_classify_fall" in targets:
        detect = targets["y_detect_fall"]
        fall_mask = targets["y_classify_fall"] >= 0
        fall_values = np.unique(detect[fall_mask]).tolist()
        nonfall_values = np.unique(detect[~fall_mask]).tolist()
        checks["fall_detection_consistency"] = {
            "values_on_fall_type_valid_windows": fall_values,
            "values_on_other_windows": nonfall_values,
            "mismatches_if_expected_0_is_fall": int(np.sum(detect[fall_mask] != 0) + np.sum(detect[~fall_mask] != 1)),
            "mismatches_if_expected_1_is_fall": int(np.sum(detect[fall_mask] != 1) + np.sum(detect[~fall_mask] != 0)),
        }

    checks.update(metadata_checks(folder.parent.parent, window, stride))
    return class_rows, task_rows, subject_rows, condition_rows, checks


def main() -> None:
    args = parse_args()
    folders = sorted(path for path in args.root.iterdir() if path.is_dir() and WINDOW_RE.fullmatch(path.name))
    if not folders:
        raise SystemExit(f"No window folders found under {args.root}")

    all_classes, all_tasks, all_subjects, all_conditions, all_checks = [], [], [], [], []
    for folder in folders:
        classes, tasks, subjects, conditions, checks = summarize_folder(folder)
        all_classes.extend(classes); all_tasks.extend(tasks); all_subjects.extend(subjects)
        all_conditions.extend(conditions); all_checks.append(checks)

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_classes).to_csv(args.output / "class_counts.csv", index=False)
    pd.DataFrame(all_tasks).to_csv(args.output / "task_counts.csv", index=False)
    pd.DataFrame(all_subjects).to_csv(args.output / "subject_counts.csv", index=False)
    pd.DataFrame(all_conditions).to_csv(args.output / "condition_counts.csv", index=False)
    (args.output / "consistency_checks.json").write_text(json.dumps(all_checks, indent=2), encoding="utf-8")

    print(pd.DataFrame(all_tasks).to_string(index=False))
    print(f"\nSaved summaries to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
