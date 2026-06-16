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
}

MODEL_SUFFIXES = {
    "cnn1conv": "CNN1Conv",
    "cnn3b3conv": "CNN3B3Conv",
    "deepconvlstm": "DeepConvLSTM",
    "lstm": "LSTM",
    "mlp": "MLP",
    "rf": "RF",
    "svm": "SVM",
    "knn": "KNN",
    "lgbm": "LightGBM",
    "lightgbm": "LightGBM",
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

    modality = "Not specified"
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
                experiment.lower() == "y_unified",
            )

    raise ValueError(f"Model not recognized in: {stem}")


def task_title(name: str) -> str:
    return TASK_TITLES.get(name, name.replace("_", " ").title())


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
        r"\footnotesize",
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
        r"\footnotesize",
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
        r"\footnotesize",
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
        r"\scriptsize",
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
        r"\footnotesize",
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

                appendix_dir = (
                    APPENDIX_TABLES_ROOT
                    / window_tag
                    / slugify(task_key)
                    / slugify(model)
                    / slugify(modality)
                )
                appendix_dir.mkdir(
                    parents=True,
                    exist_ok=True,
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

                # Appendix: complete mean ± std table for all configurations.
                (appendix_dir / "all_configurations.tex").write_text(
                    summary_table(
                        configs,
                        model,
                        task,
                        modality,
                        window,
                        stride,
                        f"tab:appendix_summary_{base}",
                    ),
                    encoding="utf-8",
                )
                generated += 1

                # Appendix: full confusion totals when binary counts exist.
                confusion = confusion_table(
                    configs,
                    model,
                    task,
                    modality,
                    window,
                    stride,
                    f"tab:appendix_confusion_{base}",
                )
                if confusion:
                    (
                        appendix_dir
                        / "all_confusion_totals.tex"
                    ).write_text(
                        confusion,
                        encoding="utf-8",
                    )
                    generated += 1

                # Repository: every per-fold/per-subject table.
                folds_dir = repository_dir / "folds"
                folds_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                for config_name, values in configs.items():
                    (
                        folds_dir
                        / f"{slugify(config_name)}.tex"
                    ).write_text(
                        per_fold_table(
                            config_name,
                            values,
                            model,
                            task,
                            modality,
                            window,
                            stride,
                            (
                                f"tab:repo_folds_{base}_"
                                f"{slugify(config_name)}"
                            ),
                        ),
                        encoding="utf-8",
                    )
                    generated += 1

                    records.append({
                        "window": window,
                        "stride": stride,
                        "task": task,
                        "task_key": task_key,
                        "model": model,
                        "modality": modality,
                        "configuration": config_name,
                        "values": values,
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

    # Main-paper tables.
    (
        MAIN_TABLES_ROOT
        / "best_models_and_configurations.tex"
    ).write_text(
        build_main_best_models_table(records),
        encoding="utf-8",
    )
    generated += 1

    (
        MAIN_TABLES_ROOT
        / "modality_ablation.tex"
    ).write_text(
        build_main_modality_ablation_table(records),
        encoding="utf-8",
    )
    generated += 1

    # One curated main-paper table for every task.
    task_names = sorted(
        {record["task"] for record in records}
    )

    for task_name in task_names:
        task_slug = slugify(task_name)

        (
            MAIN_TABLES_ROOT
            / f"{task_slug}_best_models.tex"
        ).write_text(
            build_main_task_table(
                records,
                task_name,
            ),
            encoding="utf-8",
        )
        generated += 1

        # Detailed sensor/fusion comparison for CNN1Conv + Full IMU.
        task_cnn_records = [
            record
            for record in records
            if record["task"] == task_name
            and record["model"] == "CNN1Conv"
            and record["modality"] == "Full IMU"
        ]

        if task_cnn_records:
            (
                MAIN_TABLES_ROOT
                / f"{task_slug}_cnn1conv_sensor_fusion.tex"
            ).write_text(
                build_main_task_sensor_fusion_table(
                    records,
                    task_name,
                    model_name="CNN1Conv",
                    modality_name="Full IMU",
                ),
                encoding="utf-8",
            )
            generated += 1


        # Also generate one explicit main-paper table per window duration.
        task_windows = sorted(
            {
                record["window"]
                for record in records
                if record["task"] == task_name
            },
            key=float,
        )

        for window_value in task_windows:
            window_dir = (
                MAIN_TABLES_ROOT
                / f"{window_value}-sec"
            )
            window_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                window_dir
                / f"{task_slug}_best_models.tex"
            ).write_text(
                build_main_task_table_for_window(
                    records,
                    task_name,
                    window_value,
                ),
                encoding="utf-8",
            )
            generated += 1

    # TP/FP/TN/FN remain specific to binary Fall Detection.
    (
        MAIN_TABLES_ROOT
        / "fall_detection_confusion_totals.tex"
    ).write_text(
        build_main_fall_confusion_table(records),
        encoding="utf-8",
    )
    generated += 1

    # Appendix cross-experiment summaries.
    (
        APPENDIX_TABLES_ROOT
        / "all_results.tex"
    ).write_text(
        long_summary_table(
            records,
            "all",
            "tab:appendix_all_results",
            (
                "Complete comparison across window durations, tasks, "
                "models, modalities, and sensor/fusion configurations. "
                "Values are mean $\\pm$ standard deviation across LOSO folds."
            ),
        ),
        encoding="utf-8",
    )
    generated += 1

    (
        APPENDIX_TABLES_ROOT
        / "best_configuration_per_model.tex"
    ).write_text(
        long_summary_table(
            records,
            "best_per_model",
            "tab:appendix_best_configuration_per_model",
            (
                "Best sensor or fusion configuration for every combination "
                "of window duration, task, model, and modality."
            ),
        ),
        encoding="utf-8",
    )
    generated += 1

    print(f"\nGenerated {generated} LaTeX table(s).")
    print(f"Main-paper tables: {MAIN_TABLES_ROOT}")
    print("Per-window main tables are written under main/<window>-sec/.")
    print(f"Appendix tables: {APPENDIX_TABLES_ROOT}")
    print(f"Repository-only tables: {REPOSITORY_TABLES_ROOT}")


if __name__ == "__main__":
    main()
