# ============================================================
# FALL LABEL REFINEMENT (Hierarchical & Combined Update)
# ============================================================
#
# Objetivo:
# Refinar automaticamente labels de queda.
# Mantém apenas as windows próximas do spike principal.
# Remove windows falsas antes/depois do impacto.
#
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
# 1. Aponta para o novo dataset organizado
TRIALS_ROOT = Path("IPqM-Fall/organized")
META_FILE = "window_labels.csv"

FS = 90

# Região considerada queda ao redor do pico de impacto
PRE_SECONDS = 0.5
POST_SECONDS = 0.8

# Labels de queda
FALL_LABELS = {
    "FALL_1", "FALL_2", "FALL_3", "FALL_5", "FALL_6",
}

# Label aplicada quando remove falso positivo (transição para o chão, etc)
NON_FALL_LABEL = "UNKNOWN"

# ============================================================
# LOAD META
# ============================================================
meta = pd.read_csv(META_FILE)

# ============================================================
# HELPERS
# ============================================================

def load_trial(path):
    return pd.read_parquet(path).reset_index(drop=True)

def compute_combined_signal(df):
    """
    Computa a magnitude do impacto somando Força G + Velocidade Angular.
    Agora tira partido do ficheiro `_combined.parquet` que já tem tudo alinhado!
    """
    # 1. Aceleração (Aproveita a coluna 'amag' se existir)
    if 'amag' in df.columns:
        acc_mag = df['amag'].values
    else:
        acc_cols = [c for c in df.columns if c.lower() in ["ax", "ay", "az"]] or df.columns[:3]
        acc = df[acc_cols].astype(np.float32).fillna(0)
        acc_mag = np.sqrt(np.square(acc).sum(axis=1))

    # 2. Giroscópio
    gyro_cols = [c for c in df.columns if c.lower() in ["wx", "wy", "wz"]]
    if len(gyro_cols) == 0:
        gyro_cols = df.columns[3:6]  # Fallback
        
    gyro = df[gyro_cols].astype(np.float32).fillna(0)
    gyro_mag = np.sqrt(np.square(gyro).sum(axis=1))

    # 3. Sinal Combinado
    signal = acc_mag + gyro_mag

    # Sanitização final (remove inf/nan raros)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    return signal

# ============================================================
# PROCESSAMENTO
# ============================================================

updated_trials = []

# Como o `file` agora é o caminho relativo (ex: ID1/CHEST/FALL_1.parquet)
# garantimos que processamos cada ficheiro físico único.
trial_files = sorted(meta["file"].unique())

print("\n======================================")
print("A INICIAR REFINAMENTO DE QUEDAS")
print("======================================\n")

for trial_file in tqdm(trial_files, desc="A avaliar ficheiros"):

    trial_meta = meta[meta["file"] == trial_file].copy()

    # Ignora ficheiros que não são de queda
    has_fall = trial_meta["label"].isin(FALL_LABELS).any()
    if not has_fall:
        updated_trials.append(trial_meta)
        continue

    # ========================================================
    # LOAD TRIAL
    # ========================================================
    trial_path = TRIALS_ROOT / trial_file

    try:
        trial_df = load_trial(trial_path)
    except Exception as e:
        print(f"\nERRO a carregar {trial_file}: {e}")
        updated_trials.append(trial_meta)
        continue

    # ========================================================
    # COMPUTA SINAL
    # ========================================================
    try:
        signal = compute_combined_signal(trial_df)
    except Exception as e:
        print(f"\nERRO a computar sinal em {trial_file}: {e}")
        updated_trials.append(trial_meta)
        continue

    # Proteção contra sinal inválido
    if len(signal) == 0 or np.all(signal == 0):
        updated_trials.append(trial_meta)
        continue

    # ========================================================
    # PEAK GLOBAL (Encontra o momento exato do impacto no chão)
    # ========================================================
    peak_idx = int(np.argmax(signal))

    peak_start = max(0, int(peak_idx - PRE_SECONDS * FS))
    peak_end = min(len(signal), int(peak_idx + POST_SECONDS * FS))

    # ========================================================
    # REFINE WINDOWS
    # ========================================================
    for idx, row in trial_meta.iterrows():
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])
        original_label = row["label"]

        if original_label in FALL_LABELS:
            # Verifica se a janela sobrepõe-se à região do impacto
            overlap = not (end_idx < peak_start or start_idx > peak_end)

            if overlap:
                trial_meta.loc[idx, "label"] = original_label  # Mantém queda real
            else:
                trial_meta.loc[idx, "label"] = NON_FALL_LABEL  # Falso positivo

            trial_meta.loc[idx, "reviewed"] = True

    updated_trials.append(trial_meta)

# ============================================================
# SAVE
# ============================================================

final_meta = pd.concat(updated_trials, ignore_index=True)

# Mantém a mesma ordem do ficheiro original para ficar organizado
final_meta = final_meta.sort_values(by=['subject_id', 'sensor_pos', 'file', 'start_idx'])

OUTPUT_FILE = "window_labels_refined.csv"
final_meta.to_csv(OUTPUT_FILE, index=False)

# ============================================================
# STATS
# ============================================================

before_fall = meta["label"].isin(FALL_LABELS).sum()
after_fall = final_meta["label"].isin(FALL_LABELS).sum()
removed = before_fall - after_fall

print("\n======================================")
print("REFINAMENTO FINALIZADO")
print("======================================")
print(f"Ficheiro guardado como : {OUTPUT_FILE}")
print("")
print(f"Janelas de Queda originais : {before_fall}")
print(f"Janelas de Queda refinadas : {after_fall}")
print(f"Falsos positivos removidos : {removed}")
print("======================================")