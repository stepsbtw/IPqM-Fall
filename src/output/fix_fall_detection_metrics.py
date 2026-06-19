from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

WINDOW_RE = re.compile(r"(?P<window>\d+(?:\.\d+)?)-sec_(?P<stride>\d+(?:\.\d+)?)-step")
METRIC_CONVENTION_VERSION = "binary-positive-label-v2"


def parse_window_from_path(path: Path) -> str | None:
    for part in path.parts:
        match = WINDOW_RE.search(part)
        if match:
            value = float(match.group("window"))
            return str(int(value)) if value.is_integer() else f"{value:g}"
    return None


def load_subject_counts(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    frame = pd.read_csv(path)
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for _, row in frame.iterrows():
        window = float(row["window_seconds"])
        window_key = str(int(window)) if window.is_integer() else f"{window:g}"
        subject = str(row["subject"])
        counts[(window_key, subject)] = {
            "fall": int(row["y_classify_fall_valid"]),
            "nonfall": int(row["y_classify_fall_ignored"]),
            "total": int(row["total_windows"]),
        }
    return counts


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def repaired_fold(fold: dict[str, Any], old_orientation: bool) -> dict[str, Any]:
    old_tp = int(fold["tp"])
    old_fp = int(fold["fp"])
    old_fn = int(fold["fn"])
    old_tn = int(fold["tn"])

    if old_orientation:
        # Old convention used Non-Fall (label 1) as positive:
        # matrix [[old_tn, old_fp], [old_fn, old_tp]].
        tp = old_tn  # Fall predicted Fall
        fn = old_fp  # Fall predicted Non-Fall
        fp = old_fn  # Non-Fall predicted Fall
        tn = old_tp  # Non-Fall predicted Non-Fall
    else:
        tp, fp, fn, tn = old_tp, old_fp, old_fn, old_tn

    total = tp + fp + fn + tn
    fall_precision = safe_div(tp, tp + fp)
    fall_recall = safe_div(tp, tp + fn)
    fall_f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    nonfall_precision = safe_div(tn, tn + fn)
    nonfall_recall = safe_div(tn, tn + fp)
    nonfall_f1 = safe_div(2 * tn, 2 * tn + fn + fp)

    output = dict(fold)
    output.update({
        "accuracy": safe_div(tp + tn, total),
        "precision": fall_precision,
        "recall": fall_recall,
        "f1": fall_f1,
        "f1_macro": (fall_f1 + nonfall_f1) / 2,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "positive_label": 0,
        "negative_label": 1,
        "positive_class": "Fall",
        "fall_precision": fall_precision,
        "fall_recall": fall_recall,
        "fall_f1": fall_f1,
        "nonfall_precision": nonfall_precision,
        "nonfall_recall": nonfall_recall,
        "nonfall_f1": nonfall_f1,
        "class_0_precision": fall_precision,
        "class_0_recall": fall_recall,
        "class_0_f1": fall_f1,
        "class_1_precision": nonfall_precision,
        "class_1_recall": nonfall_recall,
        "class_1_f1": nonfall_f1,
        "confusion_labels": [0, 1],
        "confusion_matrix": [[int(tp), int(fn)], [int(fp), int(tn)]],
        "metric_convention_version": METRIC_CONVENTION_VERSION,
    })
    return output


def summarize(entry: dict[str, Any]) -> None:
    folds = entry.get("folds", [])
    for key in list(entry):
        if key.endswith("_mean") or key.endswith("_std"):
            del entry[key]
    ignored = {
        "fold", "test_subject", "confusion_labels", "confusion_matrix",
        "per_class_counts", "positive_class",
    }
    keys = set().union(*(fold.keys() for fold in folds)) - ignored if folds else set()
    for key in sorted(keys):
        values = [fold[key] for fold in folds if key in fold]
        if not values or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            continue
        mean = sum(float(v) for v in values) / len(values)
        var = sum((float(v) - mean) ** 2 for v in values) / len(values)
        entry[f"{key}_mean"] = mean
        entry[f"{key}_std"] = var ** 0.5


def detect_orientation(data: dict[str, Any], window: str | None, counts: dict[tuple[str, str], dict[str, int]]) -> tuple[bool, dict[str, int]]:
    matches = {"old_nonfall_positive": 0, "new_fall_positive": 0, "unknown": 0}
    for entry in data.values():
        if not isinstance(entry, dict) or "folds" not in entry:
            continue
        for fold in entry["folds"]:
            subject = str(fold.get("test_subject", ""))
            info = counts.get((window or "", subject))
            if info is None:
                matches["unknown"] += 1
                continue
            positive_support = int(fold["tp"]) + int(fold["fn"])
            if positive_support == info["nonfall"]:
                matches["old_nonfall_positive"] += 1
            elif positive_support == info["fall"]:
                matches["new_fall_positive"] += 1
            else:
                matches["unknown"] += 1
    return matches["old_nonfall_positive"] > matches["new_fall_positive"], matches


def repair_file(path: Path, output_path: Path, counts: dict[tuple[str, str], dict[str, int]], overwrite: bool) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    window = parse_window_from_path(path)
    if window is None:
        # Fall back to the first fold total if this was copied out of its window directory.
        first_entry = next(entry for entry in data.values() if isinstance(entry, dict) and entry.get("folds"))
        first_total = sum(int(first_entry["folds"][0][key]) for key in ("tp", "fp", "fn", "tn"))
        candidates = {info["total"]: key[0] for key, info in counts.items() if key[1] == str(first_entry["folds"][0].get("test_subject"))}
        window = candidates.get(first_total)

    old_orientation, matches = detect_orientation(data, window, counts)
    repaired = {}
    for config_name, entry in data.items():
        if not isinstance(entry, dict) or "folds" not in entry:
            repaired[config_name] = entry
            continue
        new_entry = {k: v for k, v in entry.items() if not (k.endswith("_mean") or k.endswith("_std"))}
        new_entry["folds"] = [repaired_fold(fold, old_orientation) for fold in entry["folds"]]
        new_entry["metric_convention_version"] = METRIC_CONVENTION_VERSION
        if old_orientation:
            new_entry["metric_repair_note"] = "Converted from old Non-Fall-positive metrics to Fall-positive metrics."
        else:
            new_entry["metric_repair_note"] = "Verified as already Fall-positive and recomputed for consistency."
        summarize(new_entry)
        repaired[config_name] = new_entry

    destination = path if overwrite else output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(repaired, indent=4) + "\n", encoding="utf-8")
    return {"path": str(path), "output": str(destination), "window": window, **matches}


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair Fall Detection result JSONs so label 0/Fall is the positive class.")
    parser.add_argument("--results", type=Path, required=True, help="Folder containing results_y_detect_fall_*.json files")
    parser.add_argument("--subject-counts", type=Path, required=True, help="CSV with per-subject fall/non-fall counts")
    parser.add_argument("--output", type=Path, default=Path("fixed_results"), help="Output folder used when not overwriting")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files in place")
    args = parser.parse_args()

    counts = load_subject_counts(args.subject_counts)
    files = sorted(args.results.rglob("results_y_detect_fall_*.json"))
    if not files:
        raise SystemExit(f"No Fall Detection result files found under {args.results}")

    report = []
    for path in files:
        rel = path.relative_to(args.results)
        output_path = args.output / rel
        report.append(repair_file(path, output_path, counts, args.overwrite))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
