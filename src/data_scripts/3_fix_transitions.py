import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =========================================================
# CONFIGURAÇÃO
# =========================================================

TRIALS_ROOT = Path("IPqM-Fall/raw")
# Certifique-se de que este é o arquivo gerado após o script de quedas
META_FILE = "IPqM-Fall/windows_fall_fixed.parquet" 
OUTPUT_FILE = "IPqM-Fall/windows_mo_fixed.parquet"

FS = 90
STATIC_THRESHOLD = 0.3
SMOOTHING_WINDOW = int(FS * 0.5)

# =========================================================
# TAXONOMIA TÁTICA (Com Durações Dinâmicas Biomecânicas)
# =========================================================

TACTICAL_TRANSITIONS = {
    # Transições Rápidas / Verticais (1.2 segundos)
    "MO3R": {"pre": "STANDING-RIFLE", "trans": "STANDING-KNEELING-SHOOTING", "post": "KNEELING-SHOOTING", "duration": 1.2},
    
    # Transições Médias (1.5 segundos)
    "MO4R": {"pre": "WALKING-SWEEPING", "trans": "WALKING-KNEELING-SHOOTING", "post": "KNEELING-SHOOTING", "duration": 1.5},
    "MO6R": {"pre": "STANDING-RIFLE", "trans": "STANDING-PRONE-SHOOTING", "post": "PRONE-SHOOTING", "duration": 1.5},
    "MO7R": {"pre": "WALKING-SWEEPING", "trans": "WALKING-PRONE-SHOOTING", "post": "PRONE-SHOOTING", "duration": 1.5},
    
    # Transições Explosivas (2.0 segundos)
    "MO5R": {"pre": "RUNNING-RIFLE", "trans": "RUNNING-KNEELING-SHOOTING", "post": "KNEELING-SHOOTING", "duration": 2.0},
    "MO8R": {"pre": "RUNNING-RIFLE", "trans": "RUNNING-PRONE-SHOOTING", "post": "PRONE-SHOOTING", "duration": 2.0},
}

# =========================================================
# HELPERS
# =========================================================

def load_trial(path):
    return pd.read_parquet(path).reset_index(drop=True)

def find_settle_point(df):
    """
    Escaneamento vetorizado (C-level) para achar o fim da rotação.
    """
    wmag = df["wmag"].values

    kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
    smooth = np.convolve(wmag, kernel, mode="same")

    # Encontra as regiões de alta movimentação no giroscópio
    above = np.where(smooth > STATIC_THRESHOLD)[0]

    if len(above) == 0:
        return len(df) - 1

    # O último ponto antes do silêncio total
    return int(above[-1])

# =========================================================
# PROCESSAMENTO PRINCIPAL
# =========================================================

meta = pd.read_parquet(META_FILE)
updated_trials = []

trial_files = sorted(meta["file"].unique())

for trial_file in tqdm(trial_files, desc="Processando Transições MO"):

    trial_meta = meta[meta["file"] == trial_file].copy()

    # Filtra apenas os códigos MO que precisamos tratar
    trial_codes = trial_meta["activity_code"].dropna().unique()
    mo_codes = [c for c in trial_codes if c in TACTICAL_TRANSITIONS]

    if len(mo_codes) == 0:
        updated_trials.append(trial_meta)
        continue

    trial_path = TRIALS_ROOT / trial_file

    try:
        trial_df = load_trial(trial_path)
    except Exception as e:
        print(f"ERRO ao carregar {trial_file}: {e}")
        updated_trials.append(trial_meta)
        continue

    settle_idx = find_settle_point(trial_df)

    for mo_code in mo_codes:

        cfg = TACTICAL_TRANSITIONS[mo_code]
        duration = int(cfg["duration"] * FS)

        trans_end = settle_idx
        trans_start = max(0, trans_end - duration)

        pre_label = cfg["pre"]
        trans_label = cfg["trans"]
        post_label = cfg["post"]

        # =====================================================
        # ROTULAGEM VIA REGRA DO PONTO MÉDIO (Midpoint)
        # =====================================================

        for idx, row in trial_meta.iterrows():

            if row["activity_code"] != mo_code:
                continue  

            start = int(row["start_idx"])
            end = int(row["end_idx"])

            # Onde cai a agulha central da janela?
            mid_idx = (start + end) / 2.0

            if mid_idx < trans_start:
                trial_meta.loc[idx, "label"] = pre_label

            elif mid_idx > trans_end:
                trial_meta.loc[idx, "label"] = post_label

            else:
                trial_meta.loc[idx, "label"] = trans_label

            trial_meta.loc[idx, "reviewed"] = True

    updated_trials.append(trial_meta)

# =========================================================
# FINAL OUTPUT
# =========================================================

final_meta = pd.concat(updated_trials, ignore_index=True)

final_meta = final_meta.sort_values(
    by=["subject_id", "sensor_pos", "file", "start_idx"]
)

final_meta.to_parquet(OUTPUT_FILE, index=False)

# =========================================================
# SUMMARY
# =========================================================

print("\n==================== SUMMARY ====================")
print("Manobras Táticas Extraídas (Janelas Contínuas):\n")

trans_labels = {v["trans"] for v in TACTICAL_TRANSITIONS.values()}

for label in sorted(trans_labels):
    c = (final_meta["label"] == label).sum()
    if c > 0:
        print(f"{label}: {c}")

print("=================================================\n")
print(f"Dataset finalizado e salvo em: {OUTPUT_FILE}")