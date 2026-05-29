import numpy as np
import json
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURAÇÃO
# ============================================================
DATASET_DIR = Path("IPqM-Fall/windowed")
JSON_PATH = DATASET_DIR / "mapping.json"
REPORT_PATH = DATASET_DIR / "inspection_report.txt"

# Memória para armazenar o texto do relatório
report_lines = []

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def log(message=""):
    """Imprime no terminal e armazena na lista do relatório."""
    print(message)
    report_lines.append(message)

def format_class_name(class_list):
    """Resume a lista de strings para caber na tela do terminal."""
    if isinstance(class_list, str):
        return class_list
    elif isinstance(class_list, list):
        if len(class_list) > 3:
            return f"{class_list[0]}, {class_list[1]} ... (+{len(class_list)-2} classes)"
        else:
            return ", ".join(class_list)
    return "Desconhecido"

def analyze_head(head_name, npy_filename, map_key, label_map):
    npy_path = DATASET_DIR / npy_filename
    
    if not npy_path.exists():
        log(f"[!] Arquivo não encontrado: {npy_filename}")
        return

    # Carrega o array NumPy
    y = np.load(npy_path)
    total_windows = len(y)
    
    # Conta as ocorrências
    counts = Counter(y)
    
    # Pega o dicionário de mapeamento específico (ex: y_fall_map)
    head_map = label_map.get(map_key, {})
    
    log(f"\n{'='*60}")
    log(f" DISTRIBUIÇÃO: {head_name.upper()} ({npy_filename})")
    log(f"{'='*60}")
    log(f"Total de Janelas Sincronizadas: {total_windows}")
    log("-" * 60)
    log(f"{'ID':<5} | {'Amostras':<10} | {'% do Total':<12} | {'Classes Originais (Resumo)'}")
    log("-" * 60)

    # Ordena as chaves (colocando o -1 por último para facilitar a leitura)
    keys = sorted(list(counts.keys()), key=lambda x: 999 if x == -1 else x)

    for k in keys:
        count = counts[k]
        pct = (count / total_windows) * 100
        
        # Resgata o nome/descrição da classe do JSON
        class_desc = format_class_name(head_map.get(str(k), "Classe Ignorada"))
        
        if k == -1:
            class_desc = f"[IGNORADAS NO TREINO] {class_desc}"
            
        log(f"{k:<5} | {count:<10} | {pct:>5.1f}%       | {class_desc}")
        
    # Calcula e exibe a viabilidade de treinamento
    valid_samples = total_windows - counts.get(-1, 0)
    log("-" * 60)
    log(f"Amostras ÚTEIS para treino (excluindo -1): {valid_samples} ({((valid_samples/total_windows)*100):.1f}%)")
    log("\n")

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == "__main__":
    if not JSON_PATH.exists():
        log(f"Erro: Mapa JSON não encontrado em {JSON_PATH}")
        exit()

    with open(JSON_PATH, "r") as f:
        label_map = json.load(f)

    # Lista das cabeças da rede que queremos inspecionar
    heads_to_inspect = [
        ("Quedas (Detecção de Anomalia)", "y_fall.npy", "y_fall_map"),
        ("Postura Estática", "y_static.npy", "y_static_map"),
        ("Locomoção Dinâmica", "y_dynamic.npy", "y_dynamic_map"),
        ("Transições e Manobras Táticas", "y_transition.npy", "y_transition_map"),
        ("Status de Armamento", "y_weapon.npy", "y_weapon_map"),
        ("Taxonomia Completa (Omni-Classe)", "y_complete.npy", "y_complete_map")
    ]

    log("\nIniciando Inspeção do Dataset...")
    for name, filename, map_key in heads_to_inspect:
        analyze_head(name, filename, map_key, label_map)
        
    log("Inspeção concluída com sucesso.")

    # ============================================================
    # SALVAR RELATÓRIO EM ARQUIVO
    # ============================================================
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"\n[+] Relatório completo salvo em: {REPORT_PATH}")