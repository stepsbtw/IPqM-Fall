from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import json
import re


RESULTS_ROOT = Path("results")
TABLES_ROOT = RESULTS_ROOT / "tables"
MAIN_TABLES_ROOT = TABLES_ROOT / "main"
APPENDIX_TABLES_ROOT = TABLES_ROOT / "appendix"
REPOSITORY_TABLES_ROOT = TABLES_ROOT / "repository"

for directory in (
    MAIN_TABLES_ROOT,
    APPENDIX_TABLES_ROOT,
    REPOSITORY_TABLES_ROOT,
):
    directory.mkdir(parents=True, exist_ok=True)

NAME_MAP = {
    "CHEST": "C",
    "LEFT": "L",
    "RIGHT": "R",
    "CHEST_LEFT": "C+L",
    "CHEST_RIGHT": "C+R",
    "LEFT_RIGHT": "L+R",
    "CHEST_LEFT_RIGHT": "C+L+R",
    "ENSEMBLE_CHEST_LEFT": "Ens(C+L)",
    "ENSEMBLE_CHEST_RIGHT": "Ens(C+R)",
    "ENSEMBLE_LEFT_RIGHT": "Ens(L+R)",
    "ENSEMBLE_CHEST_LEFT_RIGHT": "Ens(C+L+R)",
}

TASK_TITLES = {
    "y_detect_fall": "Fall Detection",
    "y_classify_fall": "Fall-Type Classification",
    "y_classify_posture": "Posture Classification",
    "y_classify_movement": "Movement Classification",
    "fall_detection": "Fall Detection",
    "fall_type": "Fall-Type Classification",
    "fall": "Fall Detection",
    "fall_classify": "Fall-Type Classification",
    "posture": "Posture Classification",
    "movement": "Movement Classification",
    "native": "Unified 13-Class Classification",
    "y_unified": "Unified 13-Class Classification",
    "unified": "Unified 13-Class Classification",
    "unified_classification": "Unified 13-Class Classification",
}

MODEL_SUFFIXES = {
    "cnn1conv": "CNN1Conv",
    "cnn3b3conv": "CNN3B3Conv",
    "deepconvlstm": "DeepConvLSTM",
    "lstm": "LSTM",
    "mlp": "MLP",
    "logreg": "LOGREG",
    "logisticregression": "LOGREG",
    "rf": "RF",
    "svm": "SVM",
    "knn": "KNN",
    "lgbm": "LightGBM",
    "lightgbm": "LightGBM",
}



TASK_KEY_ALIASES = {
    # Fall Detection
    "y_detect_fall": "y_detect_fall",
    "fall_detection": "y_detect_fall",
    "fall": "y_detect_fall",

    # Fall-Type Classification
    "y_classify_fall": "y_classify_fall",
    "fall_type": "y_classify_fall",
    "fall_classify": "y_classify_fall",

    # Posture Classification
    "y_classify_posture": "y_classify_posture",
    "posture": "y_classify_posture",

    # Movement Classification
    "y_classify_movement": "y_classify_movement",
    "movement": "y_classify_movement",

    # Unified taxonomy
    "y_unified": "y_unified",
    "y_classify_unified": "y_unified",
    "classify_unified": "y_unified",
    "native": "y_unified",
    "unified": "y_unified",
    "unified_classification": "y_unified",
}


def canonical_task_key(name: str) -> str:
    return TASK_KEY_ALIASES.get(
        name.lower(),
        name.lower(),
    )


def canonical_task_title(name: str) -> str:
    return task_title(
        canonical_task_key(name)
    )


CLASS_NAMES = {
    "Fall-Type Classification": {
        0: "Backward Fall",
        1: "Frontal Fall",
        2: "Lateral Fall Left",
        3: "Lateral Fall Right",
    },
    "Posture Classification": {
        0: "Standing",
        1: "Sitting",
        2: "Kneeling",
        3: "Prone and Down",
    },
    "Movement Classification": {
        0: "Walking",
        1: "Sweeping",
        2: "Running",
        3: "Jumping",
        4: "Crawling",
    },
    "Unified 13-Class Classification": {
        0: "Backward Fall",
        1: "Frontal Fall",
        2: "Lateral Fall Left",
        3: "Lateral Fall Right",
        4: "Standing",
        5: "Sitting",
        6: "Kneeling",
        7: "Prone and Down",
        8: "Walking",
        9: "Sweeping",
        10: "Running",
        11: "Jumping",
        12: "Crawling",
    },
}

MODALITIES = {
    "full_imu": "Full IMU",
    "accelerometer": "Accelerometer",
    "gyroscope": "Gyroscope",
}

WINDOW_PATTERN = re.compile(
    r"^(?P<window>\d+(?:\.\d+)?)-sec_(?P<stride>\d+(?:\.\d+)?)-step$"
)


def latex_escape(text: str) -> str:
    for old, new in {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }.items():
        text = text.replace(old, new)
    return text


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def compact(value: str) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def parse_window(path: Path) -> tuple[str, str, str]:
    for parent in path.parents:
        match = WINDOW_PATTERN.match(parent.name)
        if match:
            return (
                parent.name,
                compact(match.group("window")),
                compact(match.group("stride")),
            )
    raise ValueError(f"Window tag not found in path: {path}")


def parse_filename(stem: str) -> tuple[str, str, str, bool, bool]:
    base = stem.removeprefix("results_")
    lower = base.lower()

    modality = "Full IMU"
    for token, display in sorted(
        MODALITIES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        suffix = f"_{token}"
        if lower.endswith(suffix):
            modality = display
            base = base[:-len(suffix)]
            lower = base.lower()
            break

    for suffix, display in sorted(
        MODEL_SUFFIXES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        marker = f"_{suffix}"
        if lower.endswith(marker):
            experiment = base[:-len(marker)]
            return (
                display,
                experiment,
                modality,
                experiment.lower().startswith("multitask_"),
                canonical_task_key(experiment) == "y_unified",
            )

    raise ValueError(f"Model not recognized in: {stem}")


def task_title(name: str) -> str:
    canonical = TASK_KEY_ALIASES.get(
        name.lower(),
        name.lower(),
    )
    return TASK_TITLES.get(
        canonical,
        canonical.replace("_", " ").title(),
    )



def normalize_modality_name(name: str) -> str:
    """
    Legacy result filenames without an explicit modality suffix correspond
    to the complete accelerometer + gyroscope representation.
    """
    normalized = str(name).strip()
    if not normalized or normalized.lower() == "not specified":
        return "Full IMU"
    return normalized


def valid_configs(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: values
        for name, values in data.items()
        if isinstance(values, dict)
        and isinstance(values.get("folds"), list)
        and values["folds"]
    }


def extract_blocks(
    data: dict[str, Any],
    experiment: str,
    is_multitask: bool,
    is_unified: bool,
) -> dict[str, dict[str, dict[str, Any]]]:
    if is_unified:
        sample = next(iter(data.values()))

        # Flat unified layout used by files such as
        # results_y_classify_unified_<model>.json:
        # configuration -> {folds: [...]}
        if isinstance(sample, dict) and "folds" in sample:
            return {"native": valid_configs(data)}

        # Structured unified layout:
        # configuration -> native / mapped
        output: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for config_name, config_values in data.items():
            native = config_values.get("native", {})
            if native.get("folds"):
                output["native"][config_name] = native
            for task_name, entry in config_values.get("mapped", {}).items():
                if isinstance(entry, dict) and entry.get("folds"):
                    output[task_name][config_name] = entry
        return dict(output)

    sample = next(iter(data.values()))
    if not is_multitask and isinstance(sample, dict) and "folds" in sample:
        return {experiment: valid_configs(data)}

    task_names = {
        task
        for config_values in data.values()
        if isinstance(config_values, dict)
        for task, values in config_values.items()
        if isinstance(values, dict) and isinstance(values.get("folds"), list)
    }

    return {
        task: {
            config_name: config_values[task]
            for config_name, config_values in data.items()
            if isinstance(config_values, dict)
            and task in config_values
            and config_values[task].get("folds")
        }
        for task in sorted(task_names)
    }


def metric_spec(sample: dict[str, Any]) -> tuple[list[tuple[str, str]], str]:
    multiclass = (
        "f1_macro_mean" in sample
        or "f1_macro" in sample
        or "precision_macro_mean" in sample
        or "precision_macro" in sample
    )
    if multiclass:
        return [
            ("accuracy", "Accuracy"),
            ("f1_macro", "F1-score"),
            ("precision_macro", "Precision"),
            ("recall_macro", "Recall"),
        ], "f1_macro"

    return [
        ("accuracy", "Accuracy"),
        ("f1", "F1-score"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ], "f1"


def mean_std(values: dict[str, Any], metric: str, best: float | None = None) -> str:
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"
    if mean_key not in values or std_key not in values:
        return "-"

    value = float(values[mean_key])
    text = f"{value:.3f} $\\pm$ {float(values[std_key]):.3f}"
    return rf"\textbf{{{text}}}" if best is not None and abs(value - best) < 1e-12 else text


def fold_value(fold: dict[str, Any], metric: str) -> str:
    return "-" if metric not in fold else f"{float(fold[metric]):.3f}"


def summary_table(
    configs: dict[str, dict[str, Any]],
    model: str,
    task: str,
    modality: str,
    window: str,
    stride: str,
    label: str,
) -> str:
    metrics, ranking_metric = metric_spec(next(iter(configs.values())))
    best = {
        metric: max(
            (
                float(values[f"{metric}_mean"])
                for values in configs.values()
                if f"{metric}_mean" in values
            ),
            default=float("-inf"),
        )
        for metric, _ in metrics
    }

    ordered = sorted(
        configs.items(),
        key=lambda item: item[1].get(f"{ranking_metric}_mean", float("-inf")),
        reverse=True,
    )

    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{{model} results for {task} using {modality}. Values are mean $\pm$ standard deviation across LOSO folds (window: {window}~s; stride: {stride}~s).}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & Accuracy & F1-score & Precision & Recall \\",
        r"\midrule",
    ]

    for config_name, values in ordered:
        row = [NAME_MAP.get(config_name, latex_escape(config_name))]
        row.extend(mean_std(values, metric, best[metric]) for metric, _ in metrics)
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def confusion_table(
    configs: dict[str, dict[str, Any]],
    model: str,
    task: str,
    modality: str,
    window: str,
    stride: str,
    label: str,
) -> str | None:
    rows = []
    for config_name, values in configs.items():
        folds = values["folds"]
        if not folds or not all(
            all(key in fold for key in ("tp", "fp", "tn", "fn"))
            for fold in folds
        ):
            continue
        rows.append((
            config_name,
            sum(int(fold["tp"]) for fold in folds),
            sum(int(fold["fp"]) for fold in folds),
            sum(int(fold["tn"]) for fold in folds),
            sum(int(fold["fn"]) for fold in folds),
        ))

    if not rows:
        return None

    lines = [
        r"\begin{table}[t]",
        rf"\caption{{Aggregated confusion-matrix counts across all LOSO folds for {model} on {task} using {modality} (window: {window}~s; stride: {stride}~s). Counts are summed across folds, not averaged.}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Configuration & TP & FP & TN & FN \\",
        r"\midrule",
    ]

    for config_name, tp, fp, tn, fn in rows:
        lines.append(
            f"{NAME_MAP.get(config_name, latex_escape(config_name))} "
            f"& {tp} & {fp} & {tn} & {fn} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def per_fold_table(
    config_name: str,
    values: dict[str, Any],
    model: str,
    task: str,
    modality: str,
    window: str,
    stride: str,
    label: str,
) -> str:
    metrics, _ = metric_spec(values)
    folds = sorted(
        values["folds"],
        key=lambda fold: (
            str(fold.get("test_subject", "")),
            int(fold.get("fold", 0)),
        ),
    )

    config_display = NAME_MAP.get(config_name, config_name)
    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{Per-subject LOSO results for {latex_escape(model)} on {latex_escape(task)}, configuration {latex_escape(config_display)}, using {latex_escape(modality)} (window: {window}~s; stride: {stride}~s).}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\begin{tabular}{rrcccc}",
        r"\toprule",
        r"Fold & Test subject & Accuracy & F1-score & Precision & Recall \\",
        r"\midrule",
    ]

    for fold in folds:
        row = [
            str(fold.get("fold", "-")),
            latex_escape(str(fold.get("test_subject", "-"))),
        ]
        row.extend(fold_value(fold, metric) for metric, _ in metrics)
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def long_summary_table(
    records: list[dict[str, Any]],
    mode: str,
    label: str,
    caption: str,
) -> str:
    if mode not in {"all", "best_per_model", "model_comparison"}:
        raise ValueError(f"Unknown summary mode: {mode}")

    selected = records

    if mode in {"best_per_model", "model_comparison"}:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)

        for record in records:
            if mode == "best_per_model":
                key = (
                    record["window"],
                    record["task"],
                    record["model"],
                    record["modality"],
                )
            else:
                key = (
                    record["window"],
                    record["task"],
                    record["modality"],
                    record["model"],
                )
            grouped[key].append(record)

        selected = []
        for group in grouped.values():
            def ranking(record: dict[str, Any]) -> float:
                values = record["values"]
                _, metric = metric_spec(values)
                return float(values.get(f"{metric}_mean", float("-inf")))
            selected.append(max(group, key=ranking))

    selected = sorted(
        selected,
        key=lambda record: (
            float(record["window"]),
            record["task"],
            record["model"],
            record["modality"],
            record["configuration"],
        ),
    )

    lines = [
        r"\begin{longtable}{lllllcccc}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}\\",
        r"\toprule",
        r"Window & Task & Model & Modality & Configuration & Accuracy & F1-score & Precision & Recall \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Window & Task & Model & Modality & Configuration & Accuracy & F1-score & Precision & Recall \\",
        r"\midrule",
        r"\endhead",
    ]

    for record in selected:
        values = record["values"]
        metrics, _ = metric_spec(values)
        formatted = {title: mean_std(values, metric) for metric, title in metrics}
        lines.append(
            " & ".join([
                f"{record['window']}~s",
                latex_escape(record["task"]),
                latex_escape(record["model"]),
                latex_escape(record["modality"]),
                NAME_MAP.get(record["configuration"], latex_escape(record["configuration"])),
                formatted["Accuracy"],
                formatted["F1-score"],
                formatted["Precision"],
                formatted["Recall"],
            ]) + r" \\"
        )

    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines)


def choose_best_record(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    def ranking(record: dict[str, Any]) -> float:
        values = record["values"]
        _, metric = metric_spec(values)
        return float(
            values.get(
                f"{metric}_mean",
                float("-inf"),
            )
        )

    return max(records, key=ranking)


def build_main_best_models_table(
    records: list[dict[str, Any]],
) -> str:
    grouped: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        grouped[
            (
                record["window"],
                record["task"],
                record["modality"],
                record["model"],
            )
        ].append(record)

    selected = [
        choose_best_record(group)
        for group in grouped.values()
    ]
    selected.sort(
        key=lambda record: (
            float(record["window"]),
            record["task"],
            record["modality"],
            record["model"],
        )
    )

    lines = [
        r"\begin{longtable}{lllllcccc}",
        r"\caption{Best-performing sensor or fusion configuration for each model, task, modality, and window duration. Values are mean $\pm$ standard deviation across LOSO folds.}",
        r"\label{tab:main_best_models}\\",
        r"\toprule",
        (
            r"Window & Task & Modality & Model & Best configuration "
            r"& Accuracy & F1-score & Precision & Recall \\"
        ),
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        (
            r"Window & Task & Modality & Model & Best configuration "
            r"& Accuracy & F1-score & Precision & Recall \\"
        ),
        r"\midrule",
        r"\endhead",
    ]

    for record in selected:
        values = record["values"]
        metrics, _ = metric_spec(values)
        formatted = {
            title: mean_std(values, metric)
            for metric, title in metrics
        }
        lines.append(
            " & ".join([
                f"{record['window']}~s",
                latex_escape(record["task"]),
                latex_escape(record["modality"]),
                latex_escape(record["model"]),
                NAME_MAP.get(
                    record["configuration"],
                    latex_escape(record["configuration"]),
                ),
                formatted["Accuracy"],
                formatted["F1-score"],
                formatted["Precision"],
                formatted["Recall"],
            ]) + r" \\"
        )

    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines)


def build_main_modality_ablation_table(
    records: list[dict[str, Any]],
) -> str:
    grouped: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        grouped[
            (
                record["window"],
                record["task"],
                record["model"],
                record["modality"],
            )
        ].append(record)

    selected = [
        choose_best_record(group)
        for group in grouped.values()
    ]
    selected.sort(
        key=lambda record: (
            float(record["window"]),
            record["task"],
            record["model"],
            record["modality"],
        )
    )

    lines = [
        r"\begin{longtable}{lllllcccc}",
        r"\caption{Sensor-modality ablation using the best sensor or fusion configuration of each model. Values are mean $\pm$ standard deviation across LOSO folds.}",
        r"\label{tab:main_modality_ablation}\\",
        r"\toprule",
        (
            r"Window & Task & Model & Modality & Best configuration "
            r"& Accuracy & F1-score & Precision & Recall \\"
        ),
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        (
            r"Window & Task & Model & Modality & Best configuration "
            r"& Accuracy & F1-score & Precision & Recall \\"
        ),
        r"\midrule",
        r"\endhead",
    ]

    for record in selected:
        values = record["values"]
        metrics, _ = metric_spec(values)
        formatted = {
            title: mean_std(values, metric)
            for metric, title in metrics
        }
        lines.append(
            " & ".join([
                f"{record['window']}~s",
                latex_escape(record["task"]),
                latex_escape(record["model"]),
                latex_escape(record["modality"]),
                NAME_MAP.get(
                    record["configuration"],
                    latex_escape(record["configuration"]),
                ),
                formatted["Accuracy"],
                formatted["F1-score"],
                formatted["Precision"],
                formatted["Recall"],
            ]) + r" \\"
        )

    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines)


def build_main_fall_confusion_table(
    records: list[dict[str, Any]],
) -> str:
    fall_records = [
        record
        for record in records
        if record["task"] == "Fall Detection"
        and all(
            key in fold
            for fold in record["values"]["folds"]
            for key in ()
        )
    ]

    grouped: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in fall_records:
        grouped[
            (
                record["window"],
                record["model"],
                record["modality"],
                record["task"],
            )
        ].append(record)

    selected = []
    for group in grouped.values():
        best = choose_best_record(group)
        folds = best["values"]["folds"]
        if folds and all(
            all(key in fold for key in ("tp", "fp", "tn", "fn"))
            for fold in folds
        ):
            selected.append(best)

    selected.sort(
        key=lambda record: (
            float(record["window"]),
            record["model"],
            record["modality"],
        )
    )

    lines = [
        r"\begin{longtable}{llllrrrr}",
        r"\caption{Aggregated Fall Detection confusion-matrix counts for the best configuration of each model. Counts are summed across LOSO folds and are not averaged.}",
        r"\label{tab:main_fall_confusion}\\",
        r"\toprule",
        r"Window & Model & Modality & Configuration & TP & FP & TN & FN \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Window & Model & Modality & Configuration & TP & FP & TN & FN \\",
        r"\midrule",
        r"\endhead",
    ]

    for record in selected:
        folds = record["values"]["folds"]
        tp = sum(int(fold["tp"]) for fold in folds)
        fp = sum(int(fold["fp"]) for fold in folds)
        tn = sum(int(fold["tn"]) for fold in folds)
        fn = sum(int(fold["fn"]) for fold in folds)

        lines.append(
            " & ".join([
                f"{record['window']}~s",
                latex_escape(record["model"]),
                latex_escape(record["modality"]),
                NAME_MAP.get(
                    record["configuration"],
                    latex_escape(record["configuration"]),
                ),
                str(tp),
                str(fp),
                str(tn),
                str(fn),
            ]) + r" \\"
        )

    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines)




def build_main_task_table(
    records: list[dict[str, Any]],
    task_name: str,
) -> str:
    task_records = [
        record
        for record in records
        if record["task"] == task_name
    ]

    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in task_records:
        grouped[
            (
                record["window"],
                record["model"],
                record["modality"],
            )
        ].append(record)

    selected = [
        choose_best_record(group)
        for group in grouped.values()
    ]
    selected.sort(
        key=lambda record: (
            float(record["window"]),
            record["model"],
            record["modality"],
        )
    )

    lines = [
        r"\begin{longtable}{llllcccc}",
        rf"\caption{{Best-performing sensor or fusion configuration for {latex_escape(task_name)}. Values are mean $\pm$ standard deviation across LOSO folds.}}",
        rf"\label{{tab:main_{slugify(task_name)}}}\\",
        r"\toprule",
        (
            r"Window & Model & Modality & Best configuration "
            r"& Accuracy & F1-score & Precision & Recall \\"
        ),
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        (
            r"Window & Model & Modality & Best configuration "
            r"& Accuracy & F1-score & Precision & Recall \\"
        ),
        r"\midrule",
        r"\endhead",
    ]

    for record in selected:
        values = record["values"]
        metrics, _ = metric_spec(values)
        formatted = {
            title: mean_std(values, metric)
            for metric, title in metrics
        }

        lines.append(
            " & ".join([
                f"{record['window']}~s",
                latex_escape(record["model"]),
                latex_escape(record["modality"]),
                NAME_MAP.get(
                    record["configuration"],
                    latex_escape(record["configuration"]),
                ),
                formatted["Accuracy"],
                formatted["F1-score"],
                formatted["Precision"],
                formatted["Recall"],
            ]) + r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{longtable}",
    ]
    return "\n".join(lines)


def build_main_task_sensor_fusion_table(
    records: list[dict[str, Any]],
    task_name: str,
    model_name: str = "CNN1Conv",
    modality_name: str = "Full IMU",
) -> str:
    selected = [
        record
        for record in records
        if record["task"] == task_name
        and record["model"] == model_name
        and record["modality"] == modality_name
    ]

    selected.sort(
        key=lambda record: (
            float(record["window"]),
            record["configuration"],
        )
    )

    windows = sorted(
        {record["window"] for record in selected},
        key=float,
    )
    configurations = sorted(
        {record["configuration"] for record in selected},
        key=lambda name: list(NAME_MAP).index(name)
        if name in NAME_MAP
        else 999,
    )

    lookup = {
        (record["window"], record["configuration"]): record
        for record in selected
    }

    columns = "l" + "cccc" * len(windows)
    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{{latex_escape(model_name)} sensor and fusion comparison for {latex_escape(task_name)} using {latex_escape(modality_name)}. Values are mean $\pm$ standard deviation across LOSO folds.}}",
        rf"\label{{tab:main_{slugify(task_name)}_{slugify(model_name)}_sensor_fusion}}",
        r"\centering",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
    ]

    first_header = ["Configuration"]
    for window in windows:
        first_header.append(
            rf"\multicolumn{{4}}{{c}}{{{window} s}}"
        )
    lines.append(" & ".join(first_header) + r" \\")

    cmidrules = []
    start = 2
    for _ in windows:
        cmidrules.append(
            rf"\cmidrule(lr){{{start}-{start + 3}}}"
        )
        start += 4
    lines.extend(cmidrules)

    second_header = ["Configuration"]
    for _ in windows:
        second_header.extend(
            ["Accuracy", "F1-score", "Precision", "Recall"]
        )
    lines.append(" & ".join(second_header) + r" \\")
    lines.append(r"\midrule")

    for config_name in configurations:
        row = [NAME_MAP.get(config_name, latex_escape(config_name))]

        for window in windows:
            record = lookup.get((window, config_name))
            if record is None:
                row.extend(["-", "-", "-", "-"])
                continue

            values = record["values"]
            metrics, _ = metric_spec(values)
            formatted = {
                title: mean_std(values, metric)
                for metric, title in metrics
            }
            row.extend([
                formatted["Accuracy"],
                formatted["F1-score"],
                formatted["Precision"],
                formatted["Recall"],
            ])

        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)




def build_main_task_table_for_window(
    records: list[dict[str, Any]],
    task_name: str,
    window_value: str,
) -> str:
    task_records = [
        record
        for record in records
        if record["task"] == task_name
        and record["window"] == window_value
    ]

    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in task_records:
        grouped[
            (
                record["model"],
                record["modality"],
            )
        ].append(record)

    selected = [
        choose_best_record(group)
        for group in grouped.values()
    ]
    selected.sort(
        key=lambda record: (
            record["model"],
            record["modality"],
        )
    )

    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{Best-performing sensor or fusion configuration for {latex_escape(task_name)} using {window_value}-s windows. Values are mean $\pm$ standard deviation across LOSO folds.}}",
        rf"\label{{tab:main_{slugify(task_name)}_{slugify(window_value)}s}}",
        r"\centering",
        r"\begin{tabular}{llllcccc}",
        r"\toprule",
        (
            r"Model & Modality & Best configuration & "
            r"Accuracy & F1-score & Precision & Recall \\"
        ),
        r"\midrule",
    ]

    for record in selected:
        values = record["values"]
        metrics, _ = metric_spec(values)
        formatted = {
            title: mean_std(values, metric)
            for metric, title in metrics
        }

        lines.append(
            " & ".join([
                latex_escape(record["model"]),
                latex_escape(record["modality"]),
                NAME_MAP.get(
                    record["configuration"],
                    latex_escape(record["configuration"]),
                ),
                formatted["Accuracy"],
                formatted["F1-score"],
                formatted["Precision"],
                formatted["Recall"],
            ]) + r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)




def aggregate_per_class_counts(
    folds: list[dict[str, Any]],
) -> dict[int, dict[str, int]]:
    totals: dict[int, dict[str, int]] = {}

    for fold in folds:
        counts = fold.get("per_class_counts", {})
        for class_id, values in counts.items():
            cid = int(class_id)
            target = totals.setdefault(
                cid,
                {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            )
            for key in ("tp", "fp", "tn", "fn"):
                target[key] += int(values.get(key, 0))

    return totals


def build_main_all_task_confusion_table(
    records: list[dict[str, Any]],
    task_name: str,
) -> str | None:
    task_records = [
        record
        for record in records
        if record["task"] == task_name
    ]

    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in task_records:
        grouped[
            (
                record["window"],
                record["model"],
                record["modality"],
            )
        ].append(record)

    selected = []
    for group in grouped.values():
        best = choose_best_record(group)
        folds = best["values"].get("folds", [])

        binary_available = (
            folds
            and all(
                all(
                    key in fold
                    for key in ("tp", "fp", "tn", "fn")
                )
                for fold in folds
            )
        )
        multiclass_available = (
            folds
            and any(
                fold.get("per_class_counts")
                for fold in folds
            )
        )

        if binary_available or multiclass_available:
            selected.append(best)

    if not selected:
        return None

    selected.sort(
        key=lambda record: (
            float(record["window"]),
            record["model"],
            record["modality"],
        )
    )

    lines = [
        r"\begin{longtable}{lllllrrrr}",
        rf"\caption{{Aggregated one-vs-rest confusion totals for {latex_escape(task_name)} using the best sensor or fusion configuration of each model. Counts are summed across LOSO folds.}}",
        rf"\label{{tab:main_{slugify(task_name)}_confusion_totals}}\\",
        r"\toprule",
        (
            r"Window & Model & Modality & Configuration & Class "
            r"& TP & FP & TN & FN \\"
        ),
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        (
            r"Window & Model & Modality & Configuration & Class "
            r"& TP & FP & TN & FN \\"
        ),
        r"\midrule",
        r"\endhead",
    ]

    for record in selected:
        folds = record["values"]["folds"]
        config_name = NAME_MAP.get(
            record["configuration"],
            latex_escape(record["configuration"]),
        )

        if all(
            all(
                key in fold
                for key in ("tp", "fp", "tn", "fn")
            )
            for fold in folds
        ):
            totals = {
                0: {
                    "tp": sum(int(f["tp"]) for f in folds),
                    "fp": sum(int(f["fp"]) for f in folds),
                    "tn": sum(int(f["tn"]) for f in folds),
                    "fn": sum(int(f["fn"]) for f in folds),
                }
            }
            class_names = {0: "Fall"}
        else:
            totals = aggregate_per_class_counts(folds)
            class_names = CLASS_NAMES.get(task_name, {})

        for class_id in sorted(totals):
            values = totals[class_id]
            class_label = class_names.get(
                class_id,
                f"Class {class_id}",
            )
            lines.append(
                " & ".join([
                    f"{record['window']}~s",
                    latex_escape(record["model"]),
                    latex_escape(record["modality"]),
                    config_name,
                    latex_escape(class_label),
                    str(values["tp"]),
                    str(values["fp"]),
                    str(values["tn"]),
                    str(values["fn"]),
                ]) + r" \\"
            )

    lines += [
        r"\bottomrule",
        r"\end{longtable}",
    ]
    return "\n".join(lines)


def build_main_model_sensor_fusion_table(
    records: list[dict[str, Any]],
    task_name: str,
    model_name: str,
    modality_name: str = "Full IMU",
) -> str:
    return build_main_task_sensor_fusion_table(
        records,
        task_name,
        model_name=model_name,
        modality_name=modality_name,
    )





def two_column_table_star(
    rows: list[list[str]],
    columns: str,
    header: list[str],
    caption: str,
    label: str,
    rows_per_table: int = 24,
    font_size: str = "",
) -> str:
    """
    Build one or more full-width table* environments for IEEE two-column mode.

    Large logical tables are split into continuation tables inside the same
    generated .tex file. The first table keeps the requested label and each
    continuation receives a unique label.
    """
    if not rows:
        return ""

    chunks = [
        rows[index:index + rows_per_table]
        for index in range(0, len(rows), rows_per_table)
    ]
    tables = []

    for chunk_index, chunk in enumerate(chunks):
        continuation = chunk_index > 0
        current_caption = (
            f"{caption} (continued)"
            if continuation
            else caption
        )
        current_label = (
            f"{label}_cont_{chunk_index + 1}"
            if continuation
            else label
        )

        lines = [
            r"\begin{table*}[t]",
            rf"\caption{{{current_caption}}}",
            rf"\label{{{current_label}}}",
            r"\centering",
        ]
        if font_size:
            lines.append(font_size)
        lines += [
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            " & ".join(header) + r" \\",
            r"\midrule",
        ]

        for row in chunk:
            lines.append(" & ".join(row) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
        tables.append("\n".join(lines))

    separator = (
        "\n\n% --------------------------------------------------\n"
        "\\clearpage\n\n"
    )
    return separator.join(tables)


def convert_longtable_to_table_stars(
    latex: str,
    rows_per_table: int = 24,
) -> str:
    """
    Convert a generated longtable into one or more table* floats.

    IEEEtran uses two-column mode, where longtable is not allowed.  The
    logical table is kept in the same .tex file but split into consecutive
    full-width table* floats when necessary.
    """
    if r"\begin{longtable}" not in latex:
        return latex

    begin_match = re.search(
        r"\\begin\{longtable\}\{([^}]*)\}",
        latex,
    )
    caption_match = re.search(
        r"\\caption\{(.*?)\}",
        latex,
        flags=re.S,
    )
    label_match = re.search(
        r"\\label\{([^}]*)\}",
        latex,
    )

    if not begin_match or not caption_match or not label_match:
        raise ValueError(
            "Could not parse generated longtable."
        )

    columns = begin_match.group(1)
    caption = " ".join(
        caption_match.group(1).split()
    )
    label = label_match.group(1)

    lines = [
        line.strip()
        for line in latex.splitlines()
        if line.strip()
    ]

    # The first row after the first top rule is the column header.
    top_index = lines.index(r"\toprule")
    header = lines[top_index + 1]

    # Data starts after the repeated longtable header when present.
    if r"\endhead" in lines:
        data_start = lines.index(r"\endhead") + 1
    else:
        first_midrule = lines.index(r"\midrule")
        data_start = first_midrule + 1

    bottom_index = lines.index(r"\bottomrule", data_start)
    rows = [
        line
        for line in lines[data_start:bottom_index]
        if line not in {
            r"\toprule",
            r"\midrule",
            r"\endfirsthead",
            r"\endhead",
        }
    ]

    chunks = [
        rows[index:index + rows_per_table]
        for index in range(0, len(rows), rows_per_table)
    ]

    tables = []
    for chunk_index, chunk in enumerate(chunks):
        continuation = chunk_index > 0
        current_caption = (
            f"{caption} (continued)"
            if continuation
            else caption
        )
        current_label = (
            f"{label}_cont_{chunk_index + 1}"
            if continuation
            else label
        )

        table_lines = [
            r"\begin{table*}[t]",
            rf"\caption{{{current_caption}}}",
            rf"\label{{{current_label}}}",
            r"\centering",
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            header,
            r"\midrule",
            *chunk,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
        tables.append("\n".join(table_lines))

    separator = (
        "\n\n% --------------------------------------------------\n"
        "\\clearpage\n\n"
    )
    return separator.join(tables)


def write_combined_tables(
    path: Path,
    tables: list[str | None],
) -> int:
    valid = [
        convert_longtable_to_table_stars(
            table.strip()
        )
        for table in tables
        if table and table.strip()
    ]
    if not valid:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    separator = (
        "\n\n% --------------------------------------------------\n"
        "\\clearpage\n\n"
    )
    path.write_text(
        separator.join(valid) + "\n",
        encoding="utf-8",
    )
    return 1


def build_repository_all_folds_file(
    configs: dict[str, dict[str, Any]],
    model: str,
    task: str,
    modality: str,
    window: str,
    stride: str,
    label_prefix: str,
) -> str:
    tables = []

    for config_name, values in sorted(
        configs.items(),
        key=lambda item: (
            list(NAME_MAP).index(item[0])
            if item[0] in NAME_MAP
            else 999,
            item[0],
        ),
    ):
        tables.append(
            per_fold_table(
                config_name,
                values,
                model,
                task,
                modality,
                window,
                stride,
                f"{label_prefix}_{slugify(config_name)}",
            )
        )

    separator = (
        "\n\n% --------------------------------------------------\n"
        "\\clearpage\n\n"
    )
    return separator.join(tables)


def build_main_task_sensor_fusion_file(
    records: list[dict[str, Any]],
    task_name: str,
    modality_name: str = "Full IMU",
) -> str | None:
    model_names = sorted({
        record["model"]
        for record in records
        if record["task"] == task_name
        and record["modality"] == modality_name
    })

    tables = [
        build_main_model_sensor_fusion_table(
            records,
            task_name,
            model_name=model_name,
            modality_name=modality_name,
        )
        for model_name in model_names
    ]
    tables = [
        table
        for table in tables
        if table and table.strip()
    ]

    if not tables:
        return None

    separator = (
        "\n\n% --------------------------------------------------\n"
        "\\clearpage\n\n"
    )
    return separator.join(tables)


def build_main_all_task_best_tables(
    records: list[dict[str, Any]],
    task_names: list[str],
) -> str:
    separator = (
        "\n\n% --------------------------------------------------\n"
        "\\clearpage\n\n"
    )
    return separator.join(
        build_main_task_table(records, task_name)
        for task_name in task_names
    )


def build_main_all_task_confusions(
    records: list[dict[str, Any]],
    task_names: list[str],
) -> str | None:
    tables = [
        build_main_all_task_confusion_table(
            records,
            task_name,
        )
        for task_name in task_names
    ]
    tables = [
        table
        for table in tables
        if table and table.strip()
    ]
    if not tables:
        return None

    separator = (
        "\n\n% --------------------------------------------------\n"
        "\\clearpage\n\n"
    )
    return separator.join(tables)


def build_main_window_file(
    records: list[dict[str, Any]],
    task_names: list[str],
    window_value: str,
) -> str:
    tables = [
        build_main_task_table_for_window(
            records,
            task_name,
            window_value,
        )
        for task_name in task_names
        if any(
            record["task"] == task_name
            and record["window"] == window_value
            for record in records
        )
    ]
    separator = (
        "\n\n% --------------------------------------------------\n"
        "\\clearpage\n\n"
    )
    return separator.join(tables)



def ranking_value(record: dict[str, Any]) -> float:
    values = record["values"]
    _, metric = metric_spec(values)
    return float(values.get(f"{metric}_mean", float("-inf")))


def best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(records, key=ranking_value) if records else None


def record_descriptor(record: dict[str, Any] | None) -> str:
    if record is None:
        return "-"
    configuration = NAME_MAP.get(
        record["configuration"],
        latex_escape(record["configuration"]),
    )
    return (
        f"{latex_escape(record['model'])}, "
        f"{latex_escape(record['modality'])}, "
        f"{configuration}"
    )


def selected_f1(record: dict[str, Any] | None) -> str:
    if record is None:
        return "-"
    metric = (
        "f1"
        if "f1_mean" in record["values"]
        else "f1_macro"
    )
    return mean_std(record["values"], metric)


def selected_recall(record: dict[str, Any] | None) -> str:
    if record is None:
        return "-"
    metric = (
        "recall"
        if "recall_mean" in record["values"]
        else "recall_macro"
    )
    return mean_std(record["values"], metric)


def natural_subject_key(value: str) -> tuple[int, str]:
    """
    Sort participant identifiers numerically:
    ID1, ID2, ..., ID9, ID10, ...
    """
    text = str(value).strip()
    numeric = text.upper().removeprefix("ID")
    if numeric.isdigit():
        return int(numeric), text
    return 10**9, text


def grouped_table_star(
    *,
    caption: str,
    label: str,
    columns: str,
    header: list[str],
    groups: list[tuple[str, list[list[str]]]],
    placement: str = "t",
    tabcolsep: str | None = None,
    arraystretch: str | None = None,
) -> str:
    """
    Build one full-width table with internal semantic divisions.

    This helper never:
    - changes the document font size;
    - creates continuation tables;
    - prints modality information implicitly.
    """
    lines = [
        rf"\begin{{table*}}[{placement}]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
    ]

    if tabcolsep is not None:
        lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    if arraystretch is not None:
        lines.append(rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}")

    lines += [
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]

    total_groups = len(groups)
    column_count = len(header)

    for group_index, (group_title, rows) in enumerate(groups):
        lines.append(
            rf"\multicolumn{{{column_count}}}{{l}}{{{group_title}}} \\"
        )
        lines.append(r"\addlinespace[1pt]")

        for row in rows:
            lines.append(" & ".join(row) + r" \\")

        if group_index < total_groups - 1:
            lines += [
                r"\addlinespace[2pt]",
                r"\midrule",
            ]

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def build_core_task_window_comparison(
    records: list[dict[str, Any]],
) -> str:
    relevant = [
        record
        for record in records
        if record.get("formulation")
        in {"task_specific", "unified_native"}
    ]
    tasks = sorted({record["task"] for record in relevant})
    windows = [
        window
        for window in ("2", "5")
        if any(record["window"] == window for record in relevant)
    ]

    rows = []
    for task in tasks:
        row = [latex_escape(task)]
        for window in windows:
            record = best_record([
                item for item in relevant
                if item["task"] == task
                and item["window"] == window
            ])
            row.extend([
                record_descriptor(record),
                selected_f1(record),
                selected_recall(record),
            ])
        rows.append(row)

    header = ["Task"]
    columns = "l"
    for window in windows:
        header += [
            f"{window}-s best model/configuration",
            "F1-score",
            "Recall",
        ]
        columns += "lcc"

    return two_column_table_star(
        rows=rows,
        columns=columns,
        header=header,
        caption=(
            "Best overall result for each operational task and window "
            "duration. Model and sensor/fusion configuration "
            "are selected by mean LOSO F1-score."
        ),
        label="tab:core_task_window_comparison",
        rows_per_table=20,
    )


def fusion_family(configuration: str) -> str:
    if configuration.startswith("ENSEMBLE_"):
        return "Late Fusion"
    if "_" in configuration:
        return "Early Fusion"
    return "Single Sensor"



def build_sensor_fusion_comparison(
    records: list[dict[str, Any]],
) -> str:
    """
    Compare sensor/fusion families, internally divided by task.
    Full IMU is the implicit default and is not printed.
    """
    relevant = [
        record
        for record in records
        if record.get("formulation")
        in {"task_specific", "unified_native"}
        and record["modality"] == "Full IMU"
    ]

    windows = sorted(
        {record["window"] for record in relevant},
        key=float,
    )
    groups: list[tuple[str, list[list[str]]]] = []

    for task_name in OPERATIONAL_TASKS:
        task_rows = []

        for window in windows:
            subset = [
                item
                for item in relevant
                if item["task"] == task_name
                and item["window"] == window
            ]
            if not subset:
                continue

            selected = {
                family: best_record([
                    item
                    for item in subset
                    if fusion_family(item["configuration"]) == family
                ])
                for family in (
                    "Single Sensor",
                    "Early Fusion",
                    "Late Fusion",
                )
            }

            task_rows.append([
                "",
                f"{window}~s",
                record_descriptor(selected["Single Sensor"]),
                selected_f1(selected["Single Sensor"]),
                record_descriptor(selected["Early Fusion"]),
                selected_f1(selected["Early Fusion"]),
                record_descriptor(selected["Late Fusion"]),
                selected_f1(selected["Late Fusion"]),
            ])

        if task_rows:
            groups.append((
                rf"\textbf{{{latex_escape(task_name)}}}",
                task_rows,
            ))

    return grouped_table_star(
        caption=(
            "Best single-sensor, Early-Fusion, and Late-Fusion result "
            "for each classification task and window duration."
        ),
        label="tab:core_sensor_fusion_comparison",
        columns="lllclclc",
        header=[
            "Task",
            "Window",
            "Best Single Sensor",
            "F1-score",
            "Best Early Fusion",
            "F1-score",
            "Best Late Fusion",
            "F1-score",
        ],
        groups=groups,
        placement="t",
    )

def build_curated_modality_ablation(
    records: list[dict[str, Any]],
) -> str | None:
    relevant = [
        record
        for record in records
        if record.get("formulation")
        in {"task_specific", "unified_native"}
        and record["modality"]
        in {"Accelerometer", "Gyroscope", "Full IMU"}
    ]
    modalities = [
        modality
        for modality in (
            "Accelerometer",
            "Gyroscope",
            "Full IMU",
        )
        if any(record["modality"] == modality for record in relevant)
    ]

    if len(modalities) < 2:
        return None

    tasks = sorted({record["task"] for record in relevant})
    windows = sorted(
        {record["window"] for record in relevant},
        key=float,
    )
    rows = []

    for task in tasks:
        for window in windows:
            row = [latex_escape(task), f"{window}~s"]
            found = False
            for modality in modalities:
                record = best_record([
                    item for item in relevant
                    if item["task"] == task
                    and item["window"] == window
                    and item["modality"] == modality
                ])
                found = found or record is not None
                row.extend([
                    record_descriptor(record),
                    selected_f1(record),
                ])
            if found:
                rows.append(row)

    header = ["Task", "Window"]
    columns = "ll"
    for modality in modalities:
        header += [f"{modality} best", "F1-score"]
        columns += "lc"

    return two_column_table_star(
        rows=rows,
        columns=columns,
        header=header,
        caption=(
            "Sensor-modality ablation using the best model and "
            "sensor/fusion configuration within each condition."
        ),
        label="tab:core_modality_ablation",
        rows_per_table=18,
    )



def build_task_specific_vs_unified(
    records: list[dict[str, Any]],
) -> str | None:
    """
    Compare task-specific learning with mapped unified outputs,
    internally divided by decomposed task.
    """
    task_specific = [
        record
        for record in records
        if record.get("formulation") == "task_specific"
    ]
    mapped = [
        record
        for record in records
        if record.get("formulation") == "unified_mapped"
    ]

    if not task_specific or not mapped:
        return None

    available_tasks = (
        {record["task"] for record in task_specific}
        & {record["task"] for record in mapped}
    )
    tasks = [
        task
        for task in OPERATIONAL_TASKS
        if task != "Unified 13-Class Classification"
        and task in available_tasks
    ]
    windows = sorted(
        {record["window"] for record in task_specific + mapped},
        key=float,
    )

    groups: list[tuple[str, list[list[str]]]] = []

    for task_name in tasks:
        task_rows = []

        for window in windows:
            specific = best_record([
                record
                for record in task_specific
                if record["task"] == task_name
                and record["window"] == window
            ])
            unified = best_record([
                record
                for record in mapped
                if record["task"] == task_name
                and record["window"] == window
            ])

            if specific is None and unified is None:
                continue

            task_rows.append([
                "",
                f"{window}~s",
                record_descriptor(specific),
                selected_f1(specific),
                record_descriptor(unified),
                selected_f1(unified),
            ])

        if task_rows:
            groups.append((
                rf"\textbf{{{latex_escape(task_name)}}}",
                task_rows,
            ))

    if not groups:
        return None

    return grouped_table_star(
        caption=(
            "Task-specific learning versus mapped outputs of the unified "
            "13-class formulation."
        ),
        label="tab:core_task_specific_vs_unified",
        columns="llllcc",
        header=[
            "Task",
            "Window",
            "Best Task-Specific Model/Configuration",
            "F1-score",
            "Best Mapped Unified Model/Configuration",
            "F1-score",
        ],
        groups=groups,
        placement="t",
    )

def build_appendix_detailed_results(
    records: list[dict[str, Any]],
) -> str:
    tables = [
        build_main_best_models_table(records),
        build_main_modality_ablation_table(records),
    ]
    for task_name in sorted({record["task"] for record in records}):
        tables.append(build_main_task_table(records, task_name))

    separator = (
        "\n\n% --------------------------------------------------\n"
        "\\clearpage\n\n"
    )
    return separator.join(
        table for table in tables
        if table and table.strip()
    )



OPERATIONAL_TASKS = [
    "Fall Detection",
    "Fall-Type Classification",
    "Posture Classification",
    "Movement Classification",
    "Unified 13-Class Classification",
]


def best_task_model_record(
    records: list[dict[str, Any]],
    task_name: str,
    model_name: str,
    window_value: str,
) -> dict[str, Any] | None:
    """
    Select the best experiment for one task/model/window combination.

    Operational task decompositions use task-specific experiments, while
    Unified 13-Class Classification uses native unified experiments.
    """
    allowed_formulations = (
        {"unified_native"}
        if task_name == "Unified 13-Class Classification"
        else {"task_specific"}
    )

    candidates = [
        record
        for record in records
        if record.get("formulation") in allowed_formulations
        and record["task"] == task_name
        and record["model"] == model_name
        and record["window"] == window_value
    ]
    return best_record(candidates)


def metric_for_record(
    record: dict[str, Any] | None,
    binary_name: str,
    multiclass_name: str,
) -> str:
    if record is None:
        return "-"
    metric = (
        binary_name
        if f"{binary_name}_mean" in record["values"]
        else multiclass_name
    )
    return mean_std(record["values"], metric)


def short_configuration(
    record: dict[str, Any] | None,
) -> str:
    if record is None:
        return "-"
    return NAME_MAP.get(
        record["configuration"],
        latex_escape(record["configuration"]),
    )



def build_main_model_comparison(
    records: list[dict[str, Any]],
) -> str:
    """
    Compare all evaluated models, internally divided by classification task.
    """
    relevant = [
        record
        for record in records
        if (
            (
                record["task"] == "Unified 13-Class Classification"
                and record.get("formulation") == "unified_native"
            )
            or (
                record["task"] != "Unified 13-Class Classification"
                and record.get("formulation") == "task_specific"
            )
        )
        and record["task"] in OPERATIONAL_TASKS
    ]

    models = sorted({record["model"] for record in relevant})
    groups: list[tuple[str, list[list[str]]]] = []

    for task_name in OPERATIONAL_TASKS:
        task_rows = []

        for model_name in models:
            r2 = best_task_model_record(
                relevant,
                task_name,
                model_name,
                "2",
            )
            r5 = best_task_model_record(
                relevant,
                task_name,
                model_name,
                "5",
            )

            if r2 is None and r5 is None:
                continue

            task_rows.append([
                "",
                latex_escape(model_name),
                short_configuration(r2),
                metric_for_record(r2, "accuracy", "accuracy"),
                metric_for_record(r2, "f1", "f1_macro"),
                short_configuration(r5),
                metric_for_record(r5, "accuracy", "accuracy"),
                metric_for_record(r5, "f1", "f1_macro"),
            ])

        if task_rows:
            groups.append((
                rf"\textbf{{{latex_escape(task_name)}}}",
                task_rows,
            ))

    return grouped_table_star(
        caption=(
            "Comparison of all evaluated models across the classification "
            "tasks. For each model and window duration, the best "
            "sensor/fusion configuration is retained. Binary F1 is reported "
            "for Fall Detection and Macro-F1 for multiclass tasks. Values "
            "are mean $\\pm$ standard deviation across LOSO folds."
        ),
        label="tab:main_all_models_comparison",
        columns="llllllll",
        header=[
            "Task",
            "Model",
            "2-s Best Configuration",
            "2-s Accuracy",
            "2-s F1",
            "5-s Best Configuration",
            "5-s Accuracy",
            "5-s F1",
        ],
        groups=groups,
        placement="t",
    )

def build_appendix_task_model_table(
    records: list[dict[str, Any]],
    task_name: str,
) -> str:
    """
    One appendix table per operational task.

    All models are rows. Each window keeps the best configuration of that model.
    Accuracy, F1, and Recall are shown so the appendix gives more detail than
    the compact main-paper comparison.
    """
    relevant = [
        record
        for record in records
        if record.get("formulation") == "task_specific"
        and record["task"] == task_name
    ]

    models = sorted(
        {record["model"] for record in relevant}
    )
    rows = []

    for model_name in models:
        r2 = best_task_model_record(
            relevant,
            task_name,
            model_name,
            "2",
        )
        r5 = best_task_model_record(
            relevant,
            task_name,
            model_name,
            "5",
        )

        if r2 is None and r5 is None:
            continue

        rows.append([
            latex_escape(model_name),
            short_configuration(r2),
            metric_for_record(r2, "accuracy", "accuracy"),
            metric_for_record(r2, "f1", "f1_macro"),
            metric_for_record(r2, "recall", "recall_macro"),
            short_configuration(r5),
            metric_for_record(r5, "accuracy", "accuracy"),
            metric_for_record(r5, "f1", "f1_macro"),
            metric_for_record(r5, "recall", "recall_macro"),
        ])

    return two_column_table_star(
        rows=rows,
        columns="lllllllll",
        header=[
            "Model",
            "2-s best configuration",
            "Accuracy",
            "F1",
            "Recall",
            "5-s best configuration",
            "Accuracy",
            "F1",
            "Recall",
        ],
        caption=(
            f"Model comparison for {latex_escape(task_name)}. "
            "For each model and window duration, the best sensor/fusion "
            "and modality configuration is retained. Values are mean "
            "$\\pm$ standard deviation across LOSO folds."
        ),
        label=f"tab:appendix_{slugify(task_name)}_models",
        rows_per_table=20,
    )


def class_names_for_task(
    task_name: str,
) -> dict[int, str]:
    if task_name == "Fall Detection":
        return {
            0: "Fall",
            1: "Non-Fall",
        }
    return CLASS_NAMES.get(task_name, {})


def aggregate_binary_counts(
    folds: list[dict[str, Any]],
) -> dict[int, dict[str, int]]:
    """
    Convert binary Fall Detection counts to one-vs-rest counts for both classes.

    Existing fold metrics use class 0 (Fall) as the positive class.
    """
    tp = sum(int(fold["tp"]) for fold in folds)
    fp = sum(int(fold["fp"]) for fold in folds)
    tn = sum(int(fold["tn"]) for fold in folds)
    fn = sum(int(fold["fn"]) for fold in folds)

    return {
        0: {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        1: {"tp": tn, "fp": fn, "tn": tp, "fn": fp},
    }


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def build_selected_error_analysis(
    records: list[dict[str, Any]],
) -> str | None:
    """
    Compact appendix error analysis.

    For each operational task, select the single strongest task-specific result
    across both windows and all models/configurations, then report class-level
    Precision, Recall, and F1 from aggregated LOSO confusion counts.
    """
    rows = []

    for task_name in OPERATIONAL_TASKS:
        selected = best_record([
            record
            for record in records
            if record.get("formulation") == "task_specific"
            and record["task"] == task_name
        ])
        if selected is None:
            continue

        folds = selected["values"].get("folds", [])
        if not folds:
            continue

        if all(
            all(
                key in fold
                for key in ("tp", "fp", "tn", "fn")
            )
            for fold in folds
        ):
            totals = aggregate_binary_counts(folds)
        else:
            totals = aggregate_per_class_counts(folds)

        if not totals:
            continue

        names = class_names_for_task(task_name)
        descriptor = (
            f"{latex_escape(selected['model'])}, "
            f"{selected['window']}~s, "
            f"{short_configuration(selected)}"
        )

        for class_id in sorted(totals):
            counts = totals[class_id]
            precision = safe_ratio(
                counts["tp"],
                counts["tp"] + counts["fp"],
            )
            recall = safe_ratio(
                counts["tp"],
                counts["tp"] + counts["fn"],
            )
            f1 = safe_ratio(
                2.0 * precision * recall,
                precision + recall,
            )

            rows.append([
                latex_escape(task_name),
                descriptor,
                latex_escape(
                    names.get(
                        class_id,
                        f"Class {class_id}",
                    )
                ),
                f"{precision:.3f}",
                f"{recall:.3f}",
                f"{f1:.3f}",
            ])

    if not rows:
        return None

    return two_column_table_star(
        rows=rows,
        columns="lllccc",
        header=[
            "Task",
            "Selected model/window/configuration",
            "Class",
            "Precision",
            "Recall",
            "F1",
        ],
        caption=(
            "Selected class-level error analysis. For each operational task, "
            "the strongest task-specific result across models, windows, "
            "modalities, and sensor/fusion configurations is retained. "
            "Metrics are computed from confusion counts aggregated across "
            "the outer LOSO folds."
        ),
        label="tab:appendix_selected_error_analysis",
        rows_per_table=24,
    )



def build_appendix_all_models_by_task(
    records: list[dict[str, Any]],
) -> str:
    """
    One logical appendix table containing all models, grouped by task.

    Each model/window pair keeps only its best sensor/fusion and modality
    configuration. The table is automatically split into continuation
    table* environments if it exceeds the configured row count.
    """
    relevant = [
        record
        for record in records
        if record.get("formulation") == "task_specific"
        and record["task"] in OPERATIONAL_TASKS
    ]

    models = sorted(
        {record["model"] for record in relevant}
    )
    rows = []

    for task_name in OPERATIONAL_TASKS:
        for model_name in models:
            r2 = best_task_model_record(
                relevant,
                task_name,
                model_name,
                "2",
            )
            r5 = best_task_model_record(
                relevant,
                task_name,
                model_name,
                "5",
            )

            if r2 is None and r5 is None:
                continue

            rows.append([
                latex_escape(task_name),
                latex_escape(model_name),
                short_configuration(r2),
                metric_for_record(r2, "accuracy", "accuracy"),
                metric_for_record(r2, "f1", "f1_macro"),
                metric_for_record(r2, "recall", "recall_macro"),
                short_configuration(r5),
                metric_for_record(r5, "accuracy", "accuracy"),
                metric_for_record(r5, "f1", "f1_macro"),
                metric_for_record(r5, "recall", "recall_macro"),
            ])

    return two_column_table_star(
        rows=rows,
        columns="llllllllll",
        header=[
            "Task",
            "Model",
            "2-s best configuration",
            "Accuracy",
            "F1",
            "Recall",
            "5-s best configuration",
            "Accuracy",
            "F1",
            "Recall",
        ],
        caption=(
            "Detailed comparison of all evaluated models grouped by "
            "operational task. For each model and window duration, the "
            "best sensor/fusion configuration is retained. "
            "Binary F1 is reported for Fall Detection and Macro-F1 for "
            "multiclass tasks. Values are mean $\\pm$ standard deviation "
            "across LOSO folds."
        ),
        label="tab:appendix_all_models_by_task",
        rows_per_table=22,
    )



def best_overall_task_record(
    records: list[dict[str, Any]],
    task_name: str,
) -> dict[str, Any] | None:
    """
    Select the strongest task-specific experiment for one task across:
    - all evaluated models;
    - both window durations;
    - all modalities;
    - all sensor and fusion configurations.

    Selection uses the experiment's mean LOSO F1 score:
    binary F1 for Fall Detection and Macro-F1 for multiclass tasks.
    """
    allowed_formulations = (
        {"unified_native"}
        if task_name == "Unified 13-Class Classification"
        else {"task_specific"}
    )

    candidates = [
        record
        for record in records
        if record.get("formulation") in allowed_formulations
        and (
            record["task"] == task_name
            or (
                task_name == "Unified 13-Class Classification"
                and canonical_task_key(
                    str(record.get("task_key", record["task"]))
                ) == "y_unified"
            )
        )
    ]
    return best_record(candidates)


def fold_metric_name(
    values: dict[str, Any],
    binary_name: str,
    multiclass_name: str,
) -> str:
    return (
        binary_name
        if any(
            binary_name in fold
            for fold in values.get("folds", [])
        )
        else multiclass_name
    )


def build_best_task_full_loso_table(
    records: list[dict[str, Any]],
    task_name: str,
) -> str | None:
    """
    Appendix table for one task.

    The globally best model/configuration is selected once using mean LOSO F1.
    The table then exposes the complete 14-subject outer-LOSO results rather
    than repeating only mean ± standard deviation.
    """
    selected = best_overall_task_record(
        records,
        task_name,
    )
    if selected is None:
        return None

    values = selected["values"]
    folds = sorted(
        values.get("folds", []),
        key=lambda fold: (
            str(fold.get("test_subject", "")),
            int(fold.get("fold", 0)),
        ),
    )
    if not folds:
        return None

    accuracy_metric = fold_metric_name(
        values,
        "accuracy",
        "accuracy",
    )
    f1_metric = fold_metric_name(
        values,
        "f1",
        "f1_macro",
    )
    precision_metric = fold_metric_name(
        values,
        "precision",
        "precision_macro",
    )
    recall_metric = fold_metric_name(
        values,
        "recall",
        "recall_macro",
    )

    rows = []
    for fold in folds:
        rows.append([
            str(fold.get("fold", "-")),
            latex_escape(
                str(fold.get("test_subject", "-"))
            ),
            fold_value(fold, accuracy_metric),
            fold_value(fold, f1_metric),
            fold_value(fold, precision_metric),
            fold_value(fold, recall_metric),
        ])

    configuration = NAME_MAP.get(
        selected["configuration"],
        latex_escape(selected["configuration"]),
    )
    descriptor = (
        f"{latex_escape(selected['model'])}; "
        f"{selected['window']}-s window; "
        f"{configuration}"
    )

    return two_column_table_star(
        rows=rows,
        columns="rrcccc",
        header=[
            "Fold",
            "Test subject",
            "Accuracy",
            "F1-score",
            "Precision",
            "Recall",
        ],
        caption=(
            f"Complete 14-subject LOSO results for the selected best "
            f"{latex_escape(task_name)} experiment ({descriptor}). "
            "The experiment is selected using the highest mean LOSO "
            "F1-score across all evaluated models, window durations, "
            "modalities, and sensor/fusion configurations."
        ),
        label=(
            f"tab:appendix_best_"
            f"{slugify(task_name)}_full_loso"
        ),
        rows_per_table=20,
    )


def build_all_best_task_full_loso_tables(
    records: list[dict[str, Any]],
) -> str:
    """
    One appendix file containing five logical task sections:
    one selected best experiment and its 14 LOSO folds per operational task.
    """
    tables = [
        build_best_task_full_loso_table(
            records,
            task_name,
        )
        for task_name in OPERATIONAL_TASKS
    ]
    tables = [
        table
        for table in tables
        if table and table.strip()
    ]

    separator = (
        "\n\n% --------------------------------------------------\n"
        "\\clearpage\n\n"
    )
    return separator.join(tables)



def deduplicate_semantic_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge duplicate result records created under legacy and current task names.

    Examples treated as the same semantic task:
    - fall_type == y_classify_fall == fall_classify
    - fall_detection == y_detect_fall == fall
    - posture == y_classify_posture
    - movement == y_classify_movement

    Records remain distinct across model, window, modality, configuration, and
    formulation. When two files describe the same experiment, the record with
    the larger number of LOSO folds is retained. Ties keep the later record,
    which favors the most recently discovered file.
    """
    unique: dict[tuple[str, ...], dict[str, Any]] = {}

    for record in records:
        canonical_key = canonical_task_key(
            str(record.get("task_key", record["task"]))
        )
        normalized = dict(record)
        normalized["task_key"] = canonical_key
        normalized["task"] = task_title(canonical_key)

        key = (
            canonical_key,
            str(normalized.get("formulation", "")),
            str(normalized["model"]),
            str(normalized["window"]),
            str(normalized["stride"]),
            str(normalized["modality"]),
            str(normalized["configuration"]),
        )

        current = unique.get(key)
        if current is None:
            unique[key] = normalized
            continue

        current_folds = len(
            current.get("values", {}).get("folds", [])
        )
        new_folds = len(
            normalized.get("values", {}).get("folds", [])
        )

        if new_folds >= current_folds:
            unique[key] = normalized

    return list(unique.values())



def find_subject_fold(
    record: dict[str, Any] | None,
    subject_id: str,
) -> dict[str, Any] | None:
    if record is None:
        return None

    for fold in record.get("values", {}).get("folds", []):
        if str(fold.get("test_subject", "")) == str(subject_id):
            return fold

    return None


def build_subject_best_tasks_table(
    records: list[dict[str, Any]],
    subject_id: str,
) -> str | None:
    """
    Build one appendix table for one held-out subject.

    For every operational task, use the globally selected best experiment
    for that task and report the metrics obtained when this subject was the
    outer LOSO test participant.
    """
    rows = []

    for task_name in OPERATIONAL_TASKS:
        selected = best_overall_task_record(
            records,
            task_name,
        )
        fold = find_subject_fold(
            selected,
            subject_id,
        )

        if selected is None or fold is None:
            continue

        values = selected["values"]

        accuracy_metric = fold_metric_name(
            values,
            "accuracy",
            "accuracy",
        )
        f1_metric = fold_metric_name(
            values,
            "f1",
            "f1_macro",
        )
        precision_metric = fold_metric_name(
            values,
            "precision",
            "precision_macro",
        )
        recall_metric = fold_metric_name(
            values,
            "recall",
            "recall_macro",
        )

        configuration = NAME_MAP.get(
            selected["configuration"],
            latex_escape(selected["configuration"]),
        )

        rows.append([
            latex_escape(task_name),
            latex_escape(selected["model"]),
            f"{selected['window']}~s",
            latex_escape(selected["modality"]),
            configuration,
            fold_value(fold, accuracy_metric),
            fold_value(fold, f1_metric),
            fold_value(fold, precision_metric),
            fold_value(fold, recall_metric),
        ])

    if not rows:
        return None

    return two_column_table_star(
        rows=rows,
        columns="lllllcccc",
        header=[
            "Task",
            "Model",
            "Window",
            "Modality",
            "Configuration",
            "Accuracy",
            "F1-score",
            "Precision",
            "Recall",
        ],
        caption=(
            f"Results for held-out subject ID {latex_escape(str(subject_id))} "
            "using the globally selected best experiment for each operational "
            "task. Each experiment is selected by the highest mean LOSO "
            "F1-score across all evaluated models, windows, modalities, and "
            "sensor/fusion configurations."
        ),
        label=(
            f"tab:appendix_subject_"
            f"{slugify(str(subject_id))}_best_tasks"
        ),
        rows_per_table=10,
    )


def build_all_subject_tables(
    records: list[dict[str, Any]],
) -> str:
    """
    Build one logical appendix table per held-out subject.

    Subject IDs are collected from the selected best task records, so the
    output follows the actual LOSO participants present in the result files.
    """
    subject_ids = set()

    for task_name in OPERATIONAL_TASKS:
        selected = best_overall_task_record(
            records,
            task_name,
        )
        if selected is None:
            continue

        for fold in selected.get("values", {}).get("folds", []):
            subject_ids.add(
                str(fold.get("test_subject", ""))
            )

    subject_ids = sorted(
        {
            subject_id
            for subject_id in subject_ids
            if subject_id
        },
        key=lambda value: (
            int(value)
            if value.isdigit()
            else value
        ),
    )

    tables = [
        build_subject_best_tasks_table(
            records,
            subject_id,
        )
        for subject_id in subject_ids
    ]
    tables = [
        table
        for table in tables
        if table and table.strip()
    ]

    separator = (
        "\n\n% --------------------------------------------------\n"
        "\\clearpage\n\n"
    )
    return separator.join(tables)




def build_compact_task_loso_table(
    records: list[dict[str, Any]],
) -> str:
    """
    Full subject-level LOSO results, internally divided by task.
    """
    groups: list[tuple[str, list[list[str]]]] = []

    for task_name in OPERATIONAL_TASKS:
        selected = best_overall_task_record(records, task_name)
        if selected is None:
            continue

        configuration = NAME_MAP.get(
            selected["configuration"],
            latex_escape(selected["configuration"]),
        )
        descriptor = (
            f"{latex_escape(selected['model'])}; "
            f"{selected['window']}-s; "
            f"{configuration}"
        )

        values = selected["values"]
        accuracy_metric = fold_metric_name(
            values,
            "accuracy",
            "accuracy",
        )
        f1_metric = fold_metric_name(
            values,
            "f1",
            "f1_macro",
        )
        precision_metric = fold_metric_name(
            values,
            "precision",
            "precision_macro",
        )
        recall_metric = fold_metric_name(
            values,
            "recall",
            "recall_macro",
        )

        folds = sorted(
            values.get("folds", []),
            key=lambda fold: natural_subject_key(
                str(fold.get("test_subject", ""))
            ),
        )

        task_rows = [
            [
                "",
                latex_escape(str(fold.get("test_subject", "-"))),
                fold_value(fold, accuracy_metric),
                fold_value(fold, f1_metric),
                fold_value(fold, precision_metric),
                fold_value(fold, recall_metric),
            ]
            for fold in folds
        ]

        groups.append((
            rf"\textbf{{{latex_escape(task_name)}}} --- {descriptor}",
            task_rows,
        ))

    return grouped_table_star(
        caption=(
            "Complete subject-level LOSO results for the globally selected "
            "best experiment of each classification task."
        ),
        label="tab:appendix_selected_tasks_full_loso",
        columns="llcccc",
        header=[
            "Task",
            "Test Subject",
            "Accuracy",
            "F1-score",
            "Precision",
            "Recall",
        ],
        groups=groups,
        placement="p",
        tabcolsep="3pt",
        arraystretch="0.88",
    )


def build_compact_subject_table(
    records: list[dict[str, Any]],
) -> str:
    """
    Selected results internally divided by held-out subject.
    """
    selected_by_task = {
        task_name: best_overall_task_record(records, task_name)
        for task_name in OPERATIONAL_TASKS
    }

    subject_ids = sorted(
        {
            str(fold.get("test_subject", ""))
            for record in selected_by_task.values()
            if record is not None
            for fold in record.get("values", {}).get("folds", [])
            if str(fold.get("test_subject", ""))
        },
        key=natural_subject_key,
    )

    groups: list[tuple[str, list[list[str]]]] = []

    for subject_id in subject_ids:
        subject_rows = []

        for task_name in OPERATIONAL_TASKS:
            selected = selected_by_task.get(task_name)
            fold = find_subject_fold(selected, subject_id)

            if selected is None or fold is None:
                continue

            values = selected["values"]
            configuration = NAME_MAP.get(
                selected["configuration"],
                latex_escape(selected["configuration"]),
            )

            subject_rows.append([
                "",
                latex_escape(task_name),
                latex_escape(selected["model"]),
                f"{selected['window']}~s",
                configuration,
                fold_value(
                    fold,
                    fold_metric_name(values, "accuracy", "accuracy"),
                ),
                fold_value(
                    fold,
                    fold_metric_name(values, "f1", "f1_macro"),
                ),
                fold_value(
                    fold,
                    fold_metric_name(
                        values,
                        "precision",
                        "precision_macro",
                    ),
                ),
                fold_value(
                    fold,
                    fold_metric_name(values, "recall", "recall_macro"),
                ),
            ])

        if subject_rows:
            groups.append((
                rf"\textbf{{{latex_escape(subject_id)}}}",
                subject_rows,
            ))

    return grouped_table_star(
        caption=(
            "Results of the globally selected best experiment for each "
            "classification task, organized by held-out subject."
        ),
        label="tab:appendix_selected_results_by_subject",
        columns="lllllcccc",
        header=[
            "Test Subject",
            "Task",
            "Model",
            "Window",
            "Configuration",
            "Accuracy",
            "F1-score",
            "Precision",
            "Recall",
        ],
        groups=groups,
        placement="p",
        tabcolsep="2.5pt",
        arraystretch="0.86",
    )

def main() -> None:
    json_files = sorted(
        path
        for path in RESULTS_ROOT.rglob("results_*.json")
        if TABLES_ROOT not in path.parents
    )

    if not json_files:
        print(f"No result files found under: {RESULTS_ROOT}")
        return

    records: list[dict[str, Any]] = []
    generated = 0

    for json_file in json_files:
        try:
            window_tag, window, stride = parse_window(json_file)
            (
                model,
                experiment,
                modality,
                is_multitask,
                is_unified,
            ) = parse_filename(json_file.stem)

            with json_file.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not data:
                continue

            blocks = extract_blocks(
                data,
                experiment,
                is_multitask,
                is_unified,
            )

            for task_key, configs in blocks.items():
                configs = valid_configs(configs)
                if not configs:
                    continue

                task = task_title(task_key)
                base = (
                    f"{slugify(task_key)}_{slugify(model)}_"
                    f"{slugify(modality)}_{window}s"
                )

                repository_dir = (
                    REPOSITORY_TABLES_ROOT
                    / window_tag
                    / slugify(task_key)
                    / slugify(model)
                    / slugify(modality)
                )
                repository_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # Exhaustive experiment-level tables are repository-only.
                # They are no longer written under appendix/, because the
                # paper appendix is curated by semantic purpose.
                generated += write_combined_tables(
                    repository_dir / "experiment_tables.tex",
                    [
                        summary_table(
                            configs,
                            model,
                            task,
                            modality,
                            window,
                            stride,
                            f"tab:repository_summary_{base}",
                        ),
                        confusion_table(
                            configs,
                            model,
                            task,
                            modality,
                            window,
                            stride,
                            f"tab:repository_confusion_{base}",
                        ),
                        build_repository_all_folds_file(
                            configs,
                            model,
                            task,
                            modality,
                            window,
                            stride,
                            f"tab:repository_folds_{base}",
                        ),
                    ],
                )

                for config_name, values in configs.items():
                    records.append({
                        "window": window,
                        "stride": stride,
                        "task": task,
                        "task_key": task_key,
                        "model": model,
                        "modality": normalize_modality_name(
                            modality
                        ),
                        "configuration": config_name,
                        "values": values,
                        "formulation": (
                            "unified_native"
                            if is_unified and task_key == "native"
                            else "unified_mapped"
                            if is_unified
                            else "multitask"
                            if is_multitask
                            else "task_specific"
                        ),
                    })

                print(
                    f"Generated: {window}s | {task} | "
                    f"{model} | {modality}"
                )

        except (
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            print(f"Skipping {json_file}: {error}")

    records = deduplicate_semantic_records(records)

    # Main paper: exactly three scientific comparison tables.
    generated += write_combined_tables(
        MAIN_TABLES_ROOT / "model_comparison.tex",
        [build_main_model_comparison(records)],
    )

    generated += write_combined_tables(
        MAIN_TABLES_ROOT / "core_sensor_fusion_comparison.tex",
        [build_sensor_fusion_comparison(records)],
    )

    task_unified = build_task_specific_vs_unified(records)
    if task_unified:
        generated += write_combined_tables(
            MAIN_TABLES_ROOT
            / "core_task_specific_vs_unified.tex",
            [task_unified],
        )

    # Paper appendix: semantic task-centered LOSO tables.
    generated += write_combined_tables(
        APPENDIX_TABLES_ROOT
        / "selected_full_loso_results.tex",
        [build_compact_task_loso_table(records)],
    )

    generated += write_combined_tables(
        APPENDIX_TABLES_ROOT
        / "selected_results_by_subject.tex",
        [build_compact_subject_table(records)],
    )

    print(f"\nGenerated {generated} LaTeX table(s).")
    print(f"Main-paper tables: {MAIN_TABLES_ROOT}")
    print("Main paper: 3 comparison files. Appendix: task-centered and subject-centered semantic files.")
    print(f"Appendix tables: {APPENDIX_TABLES_ROOT}")
    print(f"Repository-only tables: {REPOSITORY_TABLES_ROOT}")


if __name__ == "__main__":
    main()
