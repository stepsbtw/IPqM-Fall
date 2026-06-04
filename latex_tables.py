from pathlib import Path
import json

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def prettify_name(stem: str) -> str:
    return (
        stem.replace("results_", "")
        .replace("multitask_", "MT_")
        .replace("_cnn1conv", "")
        .replace("_deepconvlstm", "")
        .replace("_lstm", "")
        .replace("_mlp", "")
        .replace("_rf", "")
        .replace("_svm", "")
        .replace("_knn", "")
        .replace("_", r"\_")
    )


for json_file in sorted(RESULTS_DIR.glob("results_*.json")):

    with open(json_file, "r") as f:
        data = json.load(f)
        
    if not data:
        continue

    # Detect if the JSON is single-task or multi-task based on the structure
    sample_val = next(iter(data.values()))
    is_multitask = "folds" not in sample_val

    tasks_data = {}
    if is_multitask:
        # Multi-task format: data[model_name][task_name]
        task_names = [
            k for k in sample_val.keys() 
            if isinstance(sample_val[k], dict) and "folds" in sample_val[k]
        ]
        for t in task_names:
            tasks_data[t] = {
                model: data[model][t]
                for model in data.keys()
                if t in data[model] and "folds" in data[model][t] and len(data[model][t]["folds"]) > 0
            }
    else:
        # Single-task format: data[model_name]
        tasks_data["single"] = {
            k: v for k, v in data.items()
            if isinstance(v, dict) and "folds" in v and len(v["folds"]) > 0
        }

    for task_name, configs in tasks_data.items():
        if not configs:
            print(f"Skipping {json_file.name} - Task '{task_name}': no valid configurations found")
            continue

        sample = next(iter(configs.values()))
        multiclass = "f1_macro_mean" in sample

        if multiclass:
            metrics = [
                ("accuracy", "Accuracy"),
                ("precision_macro", "Precision"),
                ("recall_macro", "Recall"),
                ("f1_macro", "Macro-F1"),
            ]
        else:
            metrics = [
                ("accuracy", "Accuracy"),
                ("precision", "Precision"),
                ("recall", "Recall"),
                ("f1", "F1"),
            ]

        # Best values for boldface
        best = {}
        for metric, _ in metrics:
            metric_key = f"{metric}_mean"
            best[metric] = max(
                (cfg[metric_key] for cfg in configs.values() if metric_key in cfg), 
                default=0.0
            )

        # Sort by F1 / Macro-F1 descending
        ranking_metric = metrics[-1][0]
        sorted_configs = sorted(
            configs.items(),
            key=lambda x: x[1].get(f"{ranking_metric}_mean", 0.0),
            reverse=True,
        )

        lines = []

        lines.append(r"\begin{table}[t]")
        
        pretty_name = prettify_name(json_file.stem)
        if is_multitask:
            caption_text = rf"LOSO results for \texttt{{{pretty_name}}} (Task: {task_name.capitalize()})."
            label_text = f"tab:{json_file.stem.replace('results_', '')}_{task_name}"
        else:
            caption_text = rf"LOSO results for \texttt{{{pretty_name}}}."
            label_text = f"tab:{json_file.stem.replace('results_', '')}"

        lines.append(rf"\caption{{{caption_text}}}")
        lines.append(rf"\label{{{label_text}}}")
        lines.append(r"\centering")
        lines.append(r"\footnotesize")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")

        header = "Configuration"
        for _, label in metrics:
            header += f" & {label}"

        lines.append(header + r" \\")
        lines.append(r"\midrule")

        for config_name, vals in sorted_configs:

            row = [NAME_MAP.get(config_name, config_name)]

            for metric, _ in metrics:
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"

                if mean_key in vals and std_key in vals:
                    mean = vals[mean_key]
                    std = vals[std_key]
                    text = f"{mean:.3f} $\\pm$ {std:.3f}"

                    if abs(mean - best[metric]) < 1e-12:
                        text = rf"\textbf{{{text}}}"
                else:
                    text = "-"

                row.append(text)

            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        if is_multitask:
            output_file = OUTPUT_DIR / f"{json_file.stem}_{task_name}.tex"
        else:
            output_file = OUTPUT_DIR / f"{json_file.stem}.tex"

        with open(output_file, "w") as f:
            f.write("\n".join(lines))

        print(f"Generated: {output_file}")