import numpy as np

import config
from utils import train_single_task, train_unified_model


def select_modality(X, channel_indices):
    return X[:, :, channel_indices]


if __name__ == "__main__":
    if hasattr(config, "print_experiment_summary"):
        config.print_experiment_summary()

    print("Loading synchronized sensor arrays...")

    X_chest_full = np.load(
        config.WINDOWED_DATASET_DIR / "X_chest.npy"
    ).astype(np.float32, copy=False)

    X_left_full = np.load(
        config.WINDOWED_DATASET_DIR / "X_left.npy"
    ).astype(np.float32, copy=False)

    X_right_full = np.load(
        config.WINDOWED_DATASET_DIR / "X_right.npy"
    ).astype(np.float32, copy=False)

    groups_full = np.load(
        config.WINDOWED_DATASET_DIR / "groups.npy"
    )

    for experiment in config.EXPERIMENTS_TO_RUN:
        print(f"\n{'#' * 72}")
        print(f"### EXPERIMENT: {experiment}")
        print(f"{'#' * 72}")

        if experiment == "TASK_MODEL_MATRIX":
            for modality_name in config.TASK_MODALITIES_TO_RUN:
                if modality_name not in config.MODALITY_ABLATIONS:
                    raise ValueError(
                        f"Unknown modality: {modality_name}"
                    )

                channel_indices = config.MODALITY_ABLATIONS[
                    modality_name
                ]

                X_chest = select_modality(
                    X_chest_full,
                    channel_indices,
                )
                X_left = select_modality(
                    X_left_full,
                    channel_indices,
                )
                X_right = select_modality(
                    X_right_full,
                    channel_indices,
                )

                print(f"\n{'=' * 72}")
                print(f"MODALITY: {modality_name}")
                print(f"CHANNELS: {channel_indices}")
                print(
                    f"MODELS: {config.TASK_MODELS_TO_RUN}"
                )
                print(
                    f"TASKS: {list(config.SINGLE_TASK_MODELS)}"
                )
                print(f"{'=' * 72}")

                for model_type in config.TASK_MODELS_TO_RUN:
                    for task_name in config.SINGLE_TASK_MODELS:
                        train_single_task(
                            task_name=task_name,
                            model_type=model_type,
                            X_chest_full=X_chest,
                            X_left_full=X_left,
                            X_right_full=X_right,
                            groups_full=groups_full,
                            experiment_tag=modality_name,
                        )

        elif experiment == "UNIFIED_MODEL_MATRIX":
            for modality_name in config.UNIFIED_MODALITIES_TO_RUN:
                if modality_name not in config.MODALITY_ABLATIONS:
                    raise ValueError(
                        f"Unknown unified modality: {modality_name}"
                    )

                channel_indices = config.MODALITY_ABLATIONS[
                    modality_name
                ]

                X_chest = select_modality(
                    X_chest_full,
                    channel_indices,
                )
                X_left = select_modality(
                    X_left_full,
                    channel_indices,
                )
                X_right = select_modality(
                    X_right_full,
                    channel_indices,
                )

                print(f"\n{'=' * 72}")
                print("UNIFIED 13-CLASS MODEL MATRIX")
                print(f"MODALITY: {modality_name}")
                print(f"CHANNELS: {channel_indices}")
                print(
                    f"MODELS: {config.UNIFIED_MODELS_TO_RUN}"
                )
                print(f"{'=' * 72}")

                for model_type in config.UNIFIED_MODELS_TO_RUN:
                    train_unified_model(
                        model_type=model_type,
                        X_chest_full=X_chest,
                        X_left_full=X_left,
                        X_right_full=X_right,
                        groups_full=groups_full,
                        experiment_tag=modality_name,
                    )

        else:
            raise ValueError(
                f"Unknown experiment mode: {experiment}"
            )

    print("\nAll requested experiments completed.")
