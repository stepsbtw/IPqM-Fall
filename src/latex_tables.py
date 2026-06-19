from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable


WINDOW_RE = re.compile(r"^(?P<window>\d+(?:\.\d+)?)-sec_(?P<stride>\d+(?:\.\d+)?)-step$")

CONFIG_NAMES = {
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

MODEL_SUFFIXES = {
    "logisticregression": "LOGREG",
    "deepconvlstm": "DeepConvLSTM",
    "cnn3b3conv": "CNN3B3Conv",
    "lightgbm": "LightGBM",
    "cnn1conv": "CNN1Conv",
    "logreg": "LOGREG",
    "lgbm": "LightGBM",
    "lstm": "LSTM",
    "mlp": "MLP",
    "svm": "SVM",
    "knn": "KNN",
    "rf": "RF",
}

MODEL_DISPLAY = {
    "LOGREG": "Logistic Regression",
    "KNN": "$k$-NN",
    "RF": "Random Forest",
}

MODALITY_SUFFIXES = {
    "accelerometer": "Accelerometer",
    "gyroscope": "Gyroscope",
    "full_imu": "Full IMU",
}

TASK_ALIASES = {
    "y_detect_fall": "y_detect_fall",
    "fall_detection": "y_detect_fall",
    "fall": "y_detect_fall",
    "y_classify_fall": "y_classify_fall",
    "fall_type": "y_classify_fall",
    "fall_classify": "y_classify_fall",
    "y_classify_posture": "y_classify_posture",
    "posture": "y_classify_posture",
    "y_classify_movement": "y_classify_movement",
    "movement": "y_classify_movement",
    "y_unified": "y_unified",
    "y_classify_unified": "y_unified",
    "classify_unified": "y_unified",
    "unified": "y_unified",
    "unified_classification": "y_unified",
    "native": "y_unified",
}

TASK_TITLES = {
    "y_detect_fall": "Fall Detection",
    "y_classify_fall": "Fall-Type Classification",
    "y_classify_posture": "Posture Classification",
    "y_classify_movement": "Movement Classification",
    "y_unified": "Unified Classification",
}

TASK_ORDER = [
    "Fall Detection",
    "Fall-Type Classification",
    "Posture Classification",
    "Movement Classification",
    "Unified Classification",
]

MULTI_SUBSETS = [
    ("CHEST_LEFT", "ENSEMBLE_CHEST_LEFT"),
    ("CHEST_RIGHT", "ENSEMBLE_CHEST_RIGHT"),
    ("LEFT_RIGHT", "ENSEMBLE_LEFT_RIGHT"),
    ("CHEST_LEFT_RIGHT", "ENSEMBLE_CHEST_LEFT_RIGHT"),
]


@dataclass(frozen=True)
class Record:
    window: str
    stride: str
    task_key: str
    task: str
    model: str
    modality: str
    configuration: str
    formulation: str
    values: dict[str, Any]


def canonical_task(name: str) -> str:
    return TASK_ALIASES.get(name.lower(), name.lower())


def task_title(name: str) -> str:
    key = canonical_task(name)
    return TASK_TITLES.get(key, key.replace("_", " ").title())


def clean_number(value: str) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def parse_window(path: Path) -> tuple[str, str]:
    for parent in path.parents:
        match = WINDOW_RE.match(parent.name)
        if match:
            return clean_number(match.group("window")), clean_number(match.group("stride"))
    raise ValueError(f"window directory not found for {path}")


def parse_filename(stem: str) -> tuple[str, str, str, bool, bool]:
    name = stem.removeprefix("results_")
    lower = name.lower()
    modality = "Full IMU"

    for suffix, display in sorted(MODALITY_SUFFIXES.items(), key=lambda item: -len(item[0])):
        marker = f"_{suffix}"
        if lower.endswith(marker):
            name = name[:-len(marker)]
            lower = name.lower()
            modality = display
            break

    for suffix, model in sorted(MODEL_SUFFIXES.items(), key=lambda item: -len(item[0])):
        marker = f"_{suffix}"
        if lower.endswith(marker):
            experiment = name[:-len(marker)]
            key = canonical_task(experiment)
            return model, experiment, modality, experiment.lower().startswith("multitask_"), key == "y_unified"

    raise ValueError(f"model suffix not recognized in {stem}")


def valid_configs(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: values
        for name, values in data.items()
        if isinstance(values, dict) and isinstance(values.get("folds"), list) and values["folds"]
    }


def extract_blocks(
    data: dict[str, Any], experiment: str, is_multitask: bool, is_unified: bool
) -> dict[str, dict[str, dict[str, Any]]]:
    if not data:
        return {}

    sample = next(iter(data.values()))

    if is_unified:
        if isinstance(sample, dict) and "folds" in sample:
            return {"native": valid_configs(data)}

        blocks: dict[str, dict[str, dict[str, Any]]] = {}
        for config, values in data.items():
            if not isinstance(values, dict):
                continue
            native = values.get("native")
            if isinstance(native, dict) and native.get("folds"):
                blocks.setdefault("native", {})[config] = native
            for task, entry in values.get("mapped", {}).items():
                if isinstance(entry, dict) and entry.get("folds"):
                    blocks.setdefault(task, {})[config] = entry
        return blocks

    if not is_multitask and isinstance(sample, dict) and "folds" in sample:
        return {experiment: valid_configs(data)}

    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    for config, values in data.items():
        if not isinstance(values, dict):
            continue
        for task, entry in values.items():
            if isinstance(entry, dict) and entry.get("folds"):
                blocks.setdefault(task, {})[config] = entry
    return blocks


def load_records(results_root: Path) -> list[Record]:
    records: list[Record] = []

    for path in sorted(results_root.rglob("results_*.json")):
        if "tables" in path.parts:
            continue
        try:
            window, stride = parse_window(path)
            model, experiment, modality, is_multitask, is_unified = parse_filename(path.stem)
            data = json.loads(path.read_text(encoding="utf-8"))
            blocks = extract_blocks(data, experiment, is_multitask, is_unified)

            for raw_task, configs in blocks.items():
                task_key = "y_unified" if raw_task == "native" else canonical_task(raw_task)
                formulation = (
                    "unified_native"
                    if is_unified and raw_task == "native"
                    else "unified_mapped"
                    if is_unified
                    else "multitask"
                    if is_multitask
                    else "task_specific"
                )
                for configuration, values in valid_configs(configs).items():
                    records.append(
                        Record(
                            window=window,
                            stride=stride,
                            task_key=task_key,
                            task=task_title(task_key),
                            model=model,
                            modality=modality or "Full IMU",
                            configuration=configuration,
                            formulation=formulation,
                            values=values,
                        )
                    )
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            print(f"Skipping {path}: {error}")

    unique: dict[tuple[str, ...], Record] = {}
    for record in records:
        key = (
            record.task_key,
            record.formulation,
            record.model,
            record.window,
            record.stride,
            record.modality,
            record.configuration,
        )
        current = unique.get(key)
        if current is None or len(record.values.get("folds", [])) >= len(current.values.get("folds", [])):
            unique[key] = record

    return list(unique.values())


def score_metric(record: Record) -> str:
    if record.task_key == "y_detect_fall":
        # Fall Detection is operationally evaluated with Fall as the
        # positive class (label 0).  Older result files may only contain
        # generic f1_mean; fixed files include fall_f1_mean explicitly.
        if "fall_f1_mean" in record.values:
            return "fall_f1"
        if "f1_mean" in record.values:
            return "f1"
    return "f1" if "f1_mean" in record.values else "f1_macro"


def score(record: Record | None) -> float:
    if record is None:
        return float("-inf")
    return float(record.values.get(f"{score_metric(record)}_mean", float("-inf")))


def mean_std(record: Record | None, metric: str | None = None) -> str:
    if record is None:
        return "--"
    metric = metric or score_metric(record)
    mean = record.values.get(f"{metric}_mean")
    std = record.values.get(f"{metric}_std")
    if mean is None or std is None:
        return "--"
    return f"${float(mean):.3f} \\pm {float(std):.3f}$"


def mean_only(record: Record | None) -> str:
    return "--" if record is None else f"${score(record):.3f}$"


def best(records: Iterable[Record]) -> Record | None:
    items = list(records)
    return max(items, key=score) if items else None


def allowed_formulation(task: str) -> str:
    return "unified_native" if task == "Unified Classification" else "task_specific"


def select(
    records: Iterable[Record],
    *,
    task: str | None = None,
    model: str | None = None,
    window: str | None = None,
    modality: str | None = None,
    formulation: str | None = None,
    configuration: str | None = None,
) -> list[Record]:
    return [
        record
        for record in records
        if (task is None or record.task == task)
        and (model is None or record.model == model)
        and (window is None or record.window == window)
        and (modality is None or record.modality == modality)
        and (formulation is None or record.formulation == formulation)
        and (configuration is None or record.configuration == configuration)
    ]


def best_for_task(records: list[Record], task: str) -> Record | None:
    return best(select(records, task=task, formulation=allowed_formulation(task)))


def model_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def config_name(configuration: str) -> str:
    return CONFIG_NAMES.get(configuration, configuration.replace("_", "\\_"))


def config_kind(configuration: str) -> str:
    if configuration.startswith("ENSEMBLE_"):
        return "Late"
    if "_" in configuration:
        return "Early"
    return "Single"


def pipeline_name(record: Record | None) -> str:
    if record is None:
        return "--"
    kind = config_kind(record.configuration)
    short = config_name(record.configuration)
    return short if kind == "Single" else f"{kind} {short.replace('Ens(', '').rstrip(')')}"


def compact_descriptor(record: Record | None) -> str:
    if record is None:
        return "--"
    return f"{model_name(record.model)}, {config_name(record.configuration)}"


def sensor_phrase(configuration: str) -> str:
    base = configuration.removeprefix("ENSEMBLE_")
    names = {
        "CHEST": "chest",
        "LEFT": "left wrist",
        "RIGHT": "right wrist",
        "CHEST_LEFT": "chest and left wrist",
        "CHEST_RIGHT": "chest and right wrist",
        "LEFT_RIGHT": "both wrists",
        "CHEST_LEFT_RIGHT": "chest and both wrists",
    }
    return names.get(base, config_name(configuration))


def full_pipeline(record: Record) -> str:
    kind = config_kind(record.configuration)
    if kind == "Single":
        sensing = sensor_phrase(record.configuration)
    else:
        sensing = f"{kind} Fusion of {sensor_phrase(record.configuration)}"
    return f"{model_name(record.model)}, {record.window}-s window, {sensing}"


def natural_subject(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", value)
    return (int(match.group(1)), value) if match else (10**9, value)


def fold_metric(record: Record, basic: str, macro: str) -> str:
    folds = record.values.get("folds", [])
    return basic if any(basic in fold for fold in folds) else macro


def fold_value(fold: dict[str, Any], metric: str) -> str:
    return "--" if metric not in fold else f"{float(fold[metric]):.3f}"


def write_table(path: Path, content: str | None) -> int:
    if not content:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return 1


def build_best_task_configurations(records: list[Record]) -> str:
    rows = []
    for task in TASK_ORDER:
        record = best_for_task(records, task)
        if record:
            rows += [
                f"{task} & {full_pipeline(record)} & {mean_std(record)} " + r" \\",
                r"\addlinespace",
            ]
    if rows and rows[-1] == r"\addlinespace":
        rows.pop()

    return "\n".join([
        r"\begin{table*}[t]",
        r"\caption{Best observed complete pipeline for each prediction task. Fall-class F1 is reported for Fall Detection and Macro-F1 for the multiclass tasks.}",
        r"\label{tab:best_task_configurations}",
        r"\centering",
        r"\begin{tabularx}{\textwidth}{l X c}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Best observed configuration} & \textbf{F1-score} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table*}",
    ])


def build_model_comparison(records: list[Record]) -> str:
    relevant = [record for record in records if record.modality == "Full IMU"]
    models = sorted({record.model for record in relevant})
    lines = []

    for task_index, task in enumerate(TASK_ORDER):
        formulation = allowed_formulation(task)
        task_rows = []
        for model in models:
            r2 = best(select(relevant, task=task, model=model, window="2", formulation=formulation))
            r5 = best(select(relevant, task=task, model=model, window="5", formulation=formulation))
            if not r2 and not r5:
                continue
            task_rows.append(
                " & ".join([
                    task,
                    model_name(model),
                    pipeline_name(r2),
                    mean_std(r2),
                    pipeline_name(r5),
                    mean_std(r5),
                ]) + r" \\"
            )
        if task_rows:
            if lines:
                lines.append(r"\addlinespace")
            lines.extend(task_rows)

    return "\n".join([
        r"\begin{table*}[t]",
        r"\caption{Complete best-observed cross-model summary. Each entry retains the strongest sensor and fusion configuration of that model at the corresponding window duration. Fall-class F1 is reported for Fall Detection and Macro-F1 for the multiclass tasks.}",
        r"\label{tab:main_all_models_comparison}",
        r"\centering",
        r"\begin{tabularx}{\textwidth}{X l X c X c}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Model} & \textbf{2-s configuration} & \textbf{2-s F1} & \textbf{5-s configuration} & \textbf{5-s F1} \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table*}",
    ])


def build_sensor_fusion_summary(records: list[Record]) -> str:
    relevant = [record for record in records if record.modality == "Full IMU"]
    lines = []

    for task in TASK_ORDER:
        formulation = allowed_formulation(task)
        for window in ("2", "5"):
            subset = select(relevant, task=task, window=window, formulation=formulation)
            single = best(record for record in subset if config_kind(record.configuration) == "Single")
            multi = best(record for record in subset if config_kind(record.configuration) != "Single")
            if not single or not multi:
                continue
            gain = score(multi) - score(single)
            lines.append(
                " & ".join([
                    task,
                    f"{window}~s",
                    f"{model_name(single.model)}, {config_name(single.configuration)}: {mean_only(single)}",
                    f"{model_name(multi.model)}, {pipeline_name(multi)}: {mean_only(multi)}",
                    f"{gain:.3f}",
                ]) + r" \\"
            )
        if lines and task != TASK_ORDER[-1]:
            lines.append(r"\addlinespace")

    return "\n".join([
        r"\begin{table*}[t]",
        r"\caption{Best individual-position and multi-position result for each task and window duration. C, L, and R denote chest, left wrist, and right wrist, respectively.}",
        r"\label{tab:sensor_fusion_summary}",
        r"\centering",
        r"\begin{tabularx}{\textwidth}{X c X X c}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Window} & \textbf{Best individual position} & \textbf{Best multi-position pipeline} & \textbf{Gain} \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table*}",
    ])


def build_controlled_window(records: list[Record]) -> str | None:
    relevant = select(records, model="CNN1Conv", modality="Full IMU", formulation="task_specific")
    lines = []

    for task in TASK_ORDER[:-1]:
        pairs = []
        configs = sorted({record.configuration for record in select(relevant, task=task)})
        for configuration in configs:
            r2 = best(select(relevant, task=task, window="2", configuration=configuration))
            r5 = best(select(relevant, task=task, window="5", configuration=configuration))
            if r2 and r5:
                pairs.append(score(r5) - score(r2))
        if not pairs:
            continue
        lines.append(
            " & ".join([
                task.replace(" Classification", ""),
                f"{sum(delta > 0 for delta in pairs)}/{len(pairs)}",
                f"${median(pairs):.3f}$" if median(pairs) < 0 else f"{median(pairs):.3f}",
                f"$[{min(pairs):.3f},\\ {max(pairs):.3f}]$",
            ]) + r" \\"
        )

    if not lines:
        return None

    return "\n".join([
        r"\begin{table}[t]",
        r"\caption{Matched CNN1Conv comparison between 5-s and 2-s windows across the same sensor and fusion configurations. The difference is 5-s minus 2-s F1 or Macro-F1.}",
        r"\label{tab:controlled_window_comparison}",
        r"\centering",
        r"\begin{tabularx}{\columnwidth}{X c c X}",
        r"\toprule",
        r"\textbf{Task} & \textbf{5-s wins} & \textbf{Median difference} & \textbf{Range} \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ])


def build_controlled_fusion(records: list[Record]) -> str | None:
    relevant = select(records, model="CNN1Conv", modality="Full IMU", formulation="task_specific")
    lines = []

    for task in TASK_ORDER[:-1]:
        for window in ("2", "5"):
            deltas = []
            for early_config, late_config in MULTI_SUBSETS:
                early = best(select(relevant, task=task, window=window, configuration=early_config))
                late = best(select(relevant, task=task, window=window, configuration=late_config))
                if early and late:
                    deltas.append(score(late) - score(early))
            if not deltas:
                continue
            value = median(deltas)
            lines.append(
                " & ".join([
                    task.replace(" Classification", ""),
                    f"{window}~s",
                    f"{sum(delta > 0 for delta in deltas)}/{len(deltas)}",
                    f"${value:.3f}$" if value < 0 else f"{value:.3f}",
                ]) + r" \\"
            )
        if lines and task != TASK_ORDER[-2]:
            lines.append(r"\addlinespace")

    if not lines:
        return None

    return "\n".join([
        r"\begin{table}[t]",
        r"\caption{Matched CNN1Conv Late-versus-Early Fusion comparison across the same four multi-position subsets. The median difference is Late minus Early F1 or Macro-F1.}",
        r"\label{tab:controlled_fusion_comparison}",
        r"\centering",
        r"\begin{tabularx}{\columnwidth}{X c c c}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Window} & \textbf{Late wins} & \textbf{Median difference} \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ])


def build_task_specific_vs_unified(records: list[Record]) -> str | None:
    relevant = [record for record in records if record.modality == "Full IMU"]
    lines = []

    for task in TASK_ORDER[:-1]:
        for window in ("2", "5"):
            specific = best(select(relevant, task=task, window=window, formulation="task_specific"))
            mapped = best(select(relevant, task=task, window=window, formulation="unified_mapped"))
            if not specific and not mapped:
                continue
            lines.append(
                " & ".join([
                    task,
                    f"{window}~s",
                    compact_descriptor(specific),
                    mean_std(specific),
                    compact_descriptor(mapped),
                    mean_std(mapped),
                ]) + r" \\"
            )
        if lines and task != TASK_ORDER[-2]:
            lines.append(r"\addlinespace")

    if not lines:
        return None

    return "\n".join([
        r"\begin{table*}[t]",
        r"\caption{Task-specific learning versus mapped outputs of the unified formulation.}",
        r"\label{tab:core_task_specific_vs_unified}",
        r"\centering",
        r"\begin{tabularx}{\textwidth}{X c X c X c}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Window} & \textbf{Best task-specific model/configuration} & \textbf{F1-score} & \textbf{Best mapped unified model/configuration} & \textbf{F1-score} \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table*}",
    ])


def selected_records(records: list[Record]) -> dict[str, Record]:
    return {task: record for task in TASK_ORDER if (record := best_for_task(records, task)) is not None}


def fold_rows(record: Record) -> list[list[str]]:
    accuracy = fold_metric(record, "accuracy", "accuracy")
    f1 = fold_metric(record, "f1", "f1_macro")
    precision = fold_metric(record, "precision", "precision_macro")
    recall = fold_metric(record, "recall", "recall_macro")
    folds = sorted(record.values.get("folds", []), key=lambda fold: natural_subject(str(fold.get("test_subject", ""))))
    return [
        [
            str(fold.get("test_subject", "--")),
            fold_value(fold, accuracy),
            fold_value(fold, f1),
            fold_value(fold, precision),
            fold_value(fold, recall),
        ]
        for fold in folds
    ]


def build_selected_full_loso(records: list[Record]) -> str:
    selected = selected_records(records)
    lines = []

    for index, task in enumerate(TASK_ORDER):
        record = selected.get(task)
        if not record:
            continue
        descriptor = f"{model_name(record.model)}; {record.window}-s; {config_name(record.configuration)}"
        lines += [
            rf"\multicolumn{{6}}{{l}}{{\textbf{{{task}}} --- {descriptor}}} \\",
            r"\addlinespace[1pt]",
        ]
        for subject, accuracy, f1, precision, recall in fold_rows(record):
            lines.append(f" & {subject} & {accuracy} & {f1} & {precision} & {recall} " + r" \\")
        if index < len(TASK_ORDER) - 1:
            lines += [r"\addlinespace[2pt]", r"\midrule"]

    return "\n".join([
        r"\begin{table*}[p]",
        r"\caption{Complete subject-level LOSO results for the globally selected best experiment of each operational task.}",
        r"\label{tab:appendix_selected_tasks_full_loso}",
        r"\centering",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.88}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Task & Test Subject & Accuracy & F1-score & Precision & Recall \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])


def find_fold(record: Record, subject: str) -> dict[str, Any] | None:
    return next(
        (fold for fold in record.values.get("folds", []) if str(fold.get("test_subject", "")) == subject),
        None,
    )


def build_subject_part(selected: dict[str, Record], subjects: list[str], suffix: str) -> str:
    lines = []
    for subject_index, subject in enumerate(subjects):
        first = True
        for task in TASK_ORDER:
            record = selected.get(task)
            if not record:
                continue
            fold = find_fold(record, subject)
            if not fold:
                continue
            accuracy = fold_metric(record, "accuracy", "accuracy")
            f1 = fold_metric(record, "f1", "f1_macro")
            precision = fold_metric(record, "precision", "precision_macro")
            recall = fold_metric(record, "recall", "recall_macro")
            lines.append(
                " & ".join([
                    subject if first else "",
                    task,
                    model_name(record.model),
                    f"{record.window}~s",
                    config_name(record.configuration),
                    fold_value(fold, accuracy),
                    fold_value(fold, f1),
                    fold_value(fold, precision),
                    fold_value(fold, recall),
                ]) + r" \\"
            )
            first = False
        if subject_index < len(subjects) - 1:
            lines += [r"\addlinespace[1pt]", r"\midrule"]

    return "\n".join([
        r"\begin{table*}[p]",
        rf"\caption{{Selected task-pipeline results by held-out participant ({subjects[0]}--{subjects[-1]}).}}",
        rf"\label{{tab:appendix_selected_results_by_subject_{suffix}}}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lllllcccc}",
        r"\toprule",
        r"Test Subject & Task & Model & Window & Configuration & Accuracy & F1-score & Precision & Recall \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])


def build_selected_by_subject(records: list[Record]) -> str:
    selected = selected_records(records)
    subjects = sorted(
        {
            str(fold.get("test_subject", ""))
            for record in selected.values()
            for fold in record.values.get("folds", [])
            if str(fold.get("test_subject", ""))
        },
        key=natural_subject,
    )
    split = (len(subjects) + 1) // 2
    parts = [subjects[:split], subjects[split:]]
    tables = [build_subject_part(selected, part, chr(ord("a") + index)) for index, part in enumerate(parts) if part]
    return "\n\n".join(tables)


def generate_tables(records: list[Record], tables_root: Path) -> int:
    main_root = tables_root / "main"
    appendix_root = tables_root / "appendix"

    outputs = {
        main_root / "best_task_configurations.tex": build_best_task_configurations(records),
        main_root / "model_comparison.tex": build_model_comparison(records),
        main_root / "sensor_fusion_summary.tex": build_sensor_fusion_summary(records),
        main_root / "controlled_window_comparison.tex": build_controlled_window(records),
        main_root / "controlled_fusion_comparison.tex": build_controlled_fusion(records),
        main_root / "core_task_specific_vs_unified.tex": build_task_specific_vs_unified(records),
        appendix_root / "selected_full_loso_results.tex": build_selected_full_loso(records),
        appendix_root / "selected_results_by_subject.tex": build_selected_by_subject(records),
    }

    generated = sum(write_table(path, content) for path, content in outputs.items())
    print("Generated files:")
    for path, content in outputs.items():
        print(f"  {'OK' if content else 'SKIPPED'}  {path}")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the LaTeX result tables used by the IPqM-Fall paper.")
    parser.add_argument("--results", type=Path, default=Path("results"), help="root containing results_*.json files")
    parser.add_argument("--tables", type=Path, default=Path("results/tables"), help="output table directory")
    args = parser.parse_args()

    records = load_records(args.results)
    if not records:
        raise SystemExit(f"No valid result records found under {args.results}")

    generated = generate_tables(records, args.tables)
    print(f"\nGenerated {generated} paper table files from {len(records)} experiment records.")


if __name__ == "__main__":
    main()
