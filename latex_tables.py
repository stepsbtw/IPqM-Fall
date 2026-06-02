from pathlib import Path
import json

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/tables")
OUTPUT_DIR.mkdir(exist_ok=True)

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

    # Keep only actual experiment entries
    configs = {
        k: v
        for k, v in data.items()
        if isinstance(v, dict)
        and "folds" in v
        and len(v["folds"]) > 0
    }

    if not configs:
        print(f"Skipping {json_file.name}: no valid configurations found")
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
        best[metric] = max(
            cfg[f"{metric}_mean"]
            for cfg in configs.values()
            if f"{metric}_mean" in cfg
        )

    # Sort by F1 / Macro-F1 descending
    ranking_metric = metrics[-1][0]
    sorted_configs = sorted(
        configs.items(),
        key=lambda x: x[1][f"{ranking_metric}_mean"],
        reverse=True,
    )

    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(
        rf"\caption{{LOSO results for \texttt{{{prettify_name(json_file.stem)}}}.}}"
    )
    lines.append(
        rf"\label{{tab:{json_file.stem.replace('results_', '')}}}"
    )
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

            mean = vals[f"{metric}_mean"]
            std = vals[f"{metric}_std"]

            text = f"{mean:.3f} $\\pm$ {std:.3f}"

            if abs(mean - best[metric]) < 1e-12:
                text = rf"\textbf{{{text}}}"

            row.append(text)

        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output_file = OUTPUT_DIR / f"{json_file.stem}.tex"

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated: {output_file}")