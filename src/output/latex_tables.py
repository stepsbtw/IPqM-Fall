from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any


RESULTS_ROOT = Path("results")
TABLES_ROOT = RESULTS_ROOT / "tables"
TABLES_ROOT.mkdir(parents=True, exist_ok=True)


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
    "y_classify_fall": "Fall Classification",
    "y_classify_posture": "Posture Classification",
    "y_classify_movement": "Movement Classification",
    "y_detect_movement": "Movement Detection",
    "y_classify_transition": "Transition Classification",
    "fall": "Fall Detection",
    "fall_classify": "Fall Classification",
    "posture": "Posture Classification",
    "movement": "Movement Classification",
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
}


WINDOW_PATTERN = re.compile(
    r"^(?P<window>\d+(?:\.\d+)?)-sec_(?P<stride>\d+(?:\.\d+)?)-step$"
)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def compact_number(value: str) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def parse_window_parameters(json_file: Path) -> tuple[str, str, str]:
    """
    Returns:
        window_tag, window_seconds, stride_seconds
    """
    for parent in json_file.parents:
        match = WINDOW_PATTERN.match(parent.name)
        if match:
            window = compact_number(match.group("window"))
            stride = compact_number(match.group("stride"))
            return parent.name, window, stride

    raise ValueError(
        f"Could not determine window and stride from the path of {json_file}. "
        "Expected a parent directory such as '5-sec_1-step'."
    )


def parse_model_and_experiment(stem: str) -> tuple[str, str, bool]:
    """
    Examples:
        results_y_detect_fall_cnn1conv
        results_multitask_FALL_DETECT_POSTURE_cnn3b3conv
    """
    base = stem.removeprefix("results_")
    is_multitask = base.startswith("multitask_")

    for suffix, display_name in sorted(
        MODEL_SUFFIXES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        marker = f"_{suffix}"
        if base.lower().endswith(marker):
            experiment = base[: -len(marker)]
            return display_name, experiment, is_multitask

    raise ValueError(
        f"Could not identify the model in '{stem}'. "
        f"Supported suffixes: {', '.join(MODEL_SUFFIXES)}."
    )


def task_title(experiment: str, task_name: str, is_multitask: bool) -> str:
    if is_multitask:
        return TASK_TITLES.get(
            task_name,
            task_name.replace("_", " ").title(),
        )

    return TASK_TITLES.get(
        experiment,
        experiment.replace("_", " ").title(),
    )


def task_slug(experiment: str, task_name: str, is_multitask: bool) -> str:
    value = task_name if is_multitask else experiment
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def extract_tasks(
    data: dict[str, Any],
) -> tuple[bool, dict[str, dict[str, dict[str, Any]]]]:
    sample_value = next(iter(data.values()))
    is_multitask = "folds" not in sample_value

    if not is_multitask:
        configs = {
            name: values
            for name, values in data.items()
            if isinstance(values, dict)
            and isinstance(values.get("folds"), list)
            and values["folds"]
        }
        return False, {"single": configs}

    task_names = [
        task
        for task, values in sample_value.items()
        if isinstance(values, dict)
        and isinstance(values.get("folds"), list)
    ]

    tasks_data = {}
    for task in task_names:
        tasks_data[task] = {
            model_name: model_values[task]
            for model_name, model_values in data.items()
            if isinstance(model_values, dict)
            and task in model_values
            and isinstance(model_values[task], dict)
            and isinstance(model_values[task].get("folds"), list)
            and model_values[task]["folds"]
        }

    return True, tasks_data


def select_metrics(
    configs: dict[str, dict[str, Any]],
) -> tuple[list[tuple[str, str]], str]:
    sample = next(iter(configs.values()))
    multiclass = "f1_macro_mean" in sample

    if multiclass:
        metrics = [
            ("accuracy", "Accuracy"),
            # ("precision_macro", "Precision"),
            # ("recall_macro", "Recall"),
            ("f1_macro", "Macro-F1"),
        ]
        ranking_metric = "f1_macro"
    else:
        metrics = [
            ("accuracy", "Accuracy"),
            # ("precision", "Precision"),
            # ("recall", "Recall"),
            ("f1", "F1"),
        ]
        ranking_metric = "f1"

    return metrics, ranking_metric


def format_value(
    values: dict[str, Any],
    metric: str,
    best_value: float,
) -> str:
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"

    if mean_key not in values or std_key not in values:
        return "-"

    mean = float(values[mean_key])
    std = float(values[std_key])
    text = f"{mean:.3f} $\\pm$ {std:.3f}"

    if abs(mean - best_value) < 1e-12:
        return rf"\textbf{{{text}}}"

    return text


def build_table(
    *,
    configs: dict[str, dict[str, Any]],
    model_name: str,
    task_name: str,
    learning_name: str,
    window_seconds: str,
    stride_seconds: str,
    label: str,
) -> str:
    metrics, ranking_metric = select_metrics(configs)

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

    sorted_configs = sorted(
        configs.items(),
        key=lambda item: item[1].get(f"{ranking_metric}_mean", 0.0),
        reverse=True,
    )

    caption = (
        f"{model_name}: Avg. across all LOSO folds for "
        f"{learning_name} {task_name} "
        f"(window: {window_seconds}~s; stride: {stride_seconds}~s)."
    )

    column_spec = "l" + "c" * len(metrics)
    header = "Configuration" + "".join(
        f" & {metric_label}" for _, metric_label in metrics
    )

    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]

    for config_name, values in sorted_configs:
        row = [NAME_MAP.get(config_name, latex_escape(config_name))]
        row.extend(
            format_value(values, metric, best[metric])
            for metric, _ in metrics
        )
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

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

    generated = 0

    for json_file in json_files:
        try:
            window_tag, window_seconds, stride_seconds = (
                parse_window_parameters(json_file)
            )
            model_name, experiment, filename_multitask = (
                parse_model_and_experiment(json_file.stem)
            )

            with json_file.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not data:
                print(f"Skipping empty file: {json_file}")
                continue

            structure_multitask, tasks_data = extract_tasks(data)
            is_multitask = filename_multitask or structure_multitask

            model_slug = model_name.lower()
            output_dir = TABLES_ROOT / window_tag / model_slug
            output_dir.mkdir(parents=True, exist_ok=True)

            for internal_task, configs in tasks_data.items():
                if not configs:
                    print(
                        f"Skipping {json_file.name}, task '{internal_task}': "
                        "no valid configurations."
                    )
                    continue

                display_task = task_title(
                    experiment,
                    internal_task,
                    is_multitask,
                )
                slug = task_slug(
                    experiment,
                    internal_task,
                    is_multitask,
                )
                learning_name = (
                    "Multi-Task"
                    if is_multitask
                    else "Single-Task"
                )

                label = (
                    f"tab:{slug}_{model_slug}_"
                    f"{window_seconds}s_{stride_seconds}s"
                )

                table = build_table(
                    configs=configs,
                    model_name=model_name,
                    task_name=display_task,
                    learning_name=learning_name,
                    window_seconds=window_seconds,
                    stride_seconds=stride_seconds,
                    label=label,
                )

                if is_multitask:
                    experiment_slug = re.sub(
                        r"[^a-zA-Z0-9]+",
                        "_",
                        experiment.removeprefix("multitask_"),
                    ).strip("_").lower()
                    output_name = (
                        f"{experiment_slug}_{slug}_{model_slug}.tex"
                    )
                else:
                    output_name = f"{slug}_{model_slug}.tex"

                output_file = output_dir / output_name
                output_file.write_text(table, encoding="utf-8")
                generated += 1
                print(f"Generated: {output_file}")

        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            print(f"Skipping {json_file}: {error}")

    print(f"\nGenerated {generated} LaTeX table(s).")


if __name__ == "__main__":
    main()
