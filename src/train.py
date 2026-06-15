import numpy as np
import config
from utils import train_single_task, train_unified_model, run_multitask

if __name__ == "__main__":
    print("Carregando Matrizes Base dos Sensores na Memória...")
    
    X_chest_full = np.load(config.WINDOWED_DATASET_DIR / "X_chest.npy")
    X_left_full  = np.load(config.WINDOWED_DATASET_DIR / "X_left.npy")
    X_right_full = np.load(config.WINDOWED_DATASET_DIR / "X_right.npy")
    groups_full  = np.load(config.WINDOWED_DATASET_DIR / "groups.npy")

    for mode in config.EXPERIMENTS_TO_RUN:
        print(f"\n{'#'*70}")
        print(f"### TESTES: {mode}")
        print(f"{'#'*70}")

        if mode == "SINGLE":
            for schema, target_model in config.SINGLE_TASK_MODELS.items():
                train_single_task(
                    task_name=schema,
                    model_type=target_model,
                    X_chest_full=X_chest_full,
                    X_left_full=X_left_full,
                    X_right_full=X_right_full,
                    groups_full=groups_full,
                    experiment_tag="FULL_IMU",
                )

        elif mode == "UNIFIED":
            train_unified_model(
                model_type=config.UNIFIED_MODEL,
                X_chest_full=X_chest_full,
                X_left_full=X_left_full,
                X_right_full=X_right_full,
                groups_full=groups_full,
            )

        elif mode == "CLASSICAL_BASELINE":
            for modality_name in config.CLASSICAL_MODALITIES_TO_RUN:
                if modality_name not in config.MODALITY_ABLATIONS:
                    raise ValueError(
                        f"Modalidade clássica inválida: {modality_name}"
                    )

                channel_indices = config.MODALITY_ABLATIONS[modality_name]

                print(f"\n{'=' * 70}")
                print(f"### BASELINES CLÁSSICOS | MODALIDADE: {modality_name}")
                print(f"### CANAIS: {channel_indices}")
                print(f"### MODELOS: {config.CLASSICAL_MODELS_TO_RUN}")
                print(f"{'=' * 70}")

                X_chest_modality = X_chest_full[:, :, channel_indices]
                X_left_modality = X_left_full[:, :, channel_indices]
                X_right_modality = X_right_full[:, :, channel_indices]

                for classical_model in config.CLASSICAL_MODELS_TO_RUN:
                    for task_name in config.SINGLE_TASK_MODELS:
                        train_single_task(
                            task_name=task_name,
                            model_type=classical_model,
                            X_chest_full=X_chest_modality,
                            X_left_full=X_left_modality,
                            X_right_full=X_right_modality,
                            groups_full=groups_full,
                            experiment_tag=modality_name,
                        )

        elif mode == "MODALITY_ABLATION":
            for modality_name, channel_indices in config.MODALITY_ABLATIONS.items():
                print(f"\n{'=' * 70}")
                print(f"### MODALIDADE: {modality_name}")
                print(f"### CANAIS: {channel_indices}")
                print(f"{'=' * 70}")

                X_chest_modality = X_chest_full[:, :, channel_indices]
                X_left_modality = X_left_full[:, :, channel_indices]
                X_right_modality = X_right_full[:, :, channel_indices]

                for schema, target_model in config.SINGLE_TASK_MODELS.items():
                    train_single_task(
                        task_name=schema,
                        model_type=target_model,
                        X_chest_full=X_chest_modality,
                        X_left_full=X_left_modality,
                        X_right_full=X_right_modality,
                        groups_full=groups_full,
                        experiment_tag=modality_name,
                    )

        else:
            run_multitask(
                mode=mode,
                X_chest_full=X_chest_full,
                X_left_full=X_left_full,
                X_right_full=X_right_full,
                groups_full=groups_full,
            )

    print("\n[!] TESTES CONCLUIDOS.")