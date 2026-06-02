import numpy as np
import config
from utils import train_single_task, run_multitask

if __name__ == "__main__":
    print("Carregando Matrizes Base dos Sensores na Memória...")
    
    X_chest_full = np.load(config.DATASET_DIR / "X_chest.npy")
    X_left_full  = np.load(config.DATASET_DIR / "X_left.npy")
    X_right_full = np.load(config.DATASET_DIR / "X_right.npy")
    groups_full  = np.load(config.DATASET_DIR / "groups.npy")

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
                    groups_full=groups_full
                )
        else:
            run_multitask(
                mode=mode,
                X_chest_full=X_chest_full, 
                X_left_full=X_left_full, 
                X_right_full=X_right_full, 
                groups_full=groups_full
            )

    print("\n[!] TESTES CONCLUIDOS.")