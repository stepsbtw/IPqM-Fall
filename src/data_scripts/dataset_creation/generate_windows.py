import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import re

# 1. Aponta para a nova pasta hierárquica
TRIALS_ROOT = Path("IPqM-Fall/organized")

WINDOW_SEC = 2
STRIDE_SEC = 1
FS = 90

WINDOW_SIZE = WINDOW_SEC * FS
STRIDE = STRIDE_SEC * FS

rows = []

# 2. Usa rglob para varrer todas as pastas (ID1/CHEST/, etc...)
trial_files = sorted(TRIALS_ROOT.rglob("*.parquet"))

start_time = time.time()

for parquet_file in tqdm(
    trial_files,
    desc="Processing trials",
    unit="file"
):

    df = pd.read_parquet(parquet_file)

    if len(df) < WINDOW_SIZE:
        continue

    stem = parquet_file.stem
    
    # 3. Extrair ID e Sensor a partir da estrutura de pastas
    # Como o caminho é .../ID1/CHEST/ADL_1_trial1_combined.parquet:
    subject_id = parquet_file.parent.parent.name  # Pega o 'ID1'
    sensor_pos = parquet_file.parent.name         # Pega o 'CHEST'
    
    # 4. Guardar o caminho relativo para ser fácil de ler no ML
    rel_path = parquet_file.relative_to(TRIALS_ROOT).as_posix()

    # Extrair a label da atividade através do nome do ficheiro
    match = re.search(
        r"(ADL_\d+|OM_\d+|FALL_\d+)",
        stem
    )

    if match:
        initial_label = match.group(1)
    else:
        initial_label = "UNKNOWN"

    n_windows = (
        (len(df) - WINDOW_SIZE) // STRIDE
    ) + 1

    for i in range(n_windows):

        start = i * STRIDE
        end = start + WINDOW_SIZE

        # Cria um ID de janela único combinando a pessoa, sensor e ficheiro
        window_id = f"{subject_id}_{sensor_pos}_{stem}_win_{start}"

        rows.append({
            "file": rel_path,             # Ex: "ID1/CHEST/ADL_1_trial1_combined.parquet"
            "subject_id": subject_id,     # Já em coluna! Adeus parser no ML!
            "sensor_pos": sensor_pos,     # Já em coluna!
            "window_id": window_id,
            "start_idx": start,
            "end_idx": end,
            "label": initial_label,
            "reviewed": False
        })

elapsed = time.time() - start_time

meta = pd.DataFrame(rows)

# Opcional: Reordenar por ID para ficar bonito
meta = meta.sort_values(by=['subject_id', 'sensor_pos', 'file', 'start_idx'])

meta.to_csv("window_labels.csv", index=False)

print("\n========================")
print(f"Trials processed : {len(trial_files)}")
print(f"Windows created  : {len(meta)}")
print(f"Elapsed time     : {elapsed:.2f} sec")
print("========================")

print(meta.head())