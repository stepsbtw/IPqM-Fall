import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURAÇÕES DE PASTAS
# ============================================================
TRIALS_ROOT = Path("IPqM-Fall/OLD/trials_90hz")
OUTPUT_ROOT = Path("IPqM-Fall/OLD/trials_90hz_combined")

# Cria a pasta de destino se ela ainda não existir
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def process_combined_parquets():
    # Lista todos os ficheiros de aceleração
    acc_files = list(TRIALS_ROOT.glob("*_acceleration.parquet"))
    
    if not acc_files:
        print("Erro: Nenhum ficheiro de aceleração encontrado na pasta raiz.")
        return

    print(f"A iniciar a fusão e limpeza de {len(acc_files)} ficheiros...\n")
    
    sucessos = 0
    erros = 0

    for acc_path in tqdm(acc_files, desc="A processar ficheiros"):
        # Descobre o par correspondente do giroscópio
        gyro_filename = acc_path.name.replace("_acceleration", "_angular_speed")
        gyro_path = TRIALS_ROOT / gyro_filename
        
        # Ignora se faltar metade do sensor
        if not gyro_path.exists():
            erros += 1
            continue
            
        try:
            # 1. Carregar os dois ficheiros e garantir ordenação temporal
            df_acc = pd.read_parquet(acc_path).sort_values("timestamp")
            df_gyro = pd.read_parquet(gyro_path).sort_values("timestamp")
            
            # 2. O Merge Inteligente (Aproximação Temporal)
            # tolerance=25 significa que aceita até 25 unidades de tempo de diferença
            df_combined = pd.merge_asof(
                left=df_acc,
                right=df_gyro,
                on="timestamp",
                direction="nearest",
                tolerance=25
            )
            
            # 3. Tratamento de Falhas (Interpolação Linear)
            # Isto resolve os 0.2% de dados perdidos calculando a média física do movimento!
            df_combined = df_combined.interpolate(method='linear', limit_direction='both')
            
            # Remove qualquer linha teimosa nas pontas que não tenha dado para interpolar
            df_combined = df_combined.dropna()
            
            # Se após limpar tudo o ficheiro ficou vazio, ignoramos
            if len(df_combined) == 0:
                erros += 1
                continue
            
            # 4. Guardar o novo ficheiro unificado
            combined_filename = acc_path.name.replace("_acceleration", "")
            combined_path = OUTPUT_ROOT / combined_filename
            
            df_combined.to_parquet(combined_path, index=False)
            sucessos += 1
            
        except Exception as e:
            # Captura ficheiros corrompidos
            erros += 1
            pass

    print("\n" + "="*50)
    print("PROCESSO CONCLUÍDO!")
    print("="*50)
    print(f"Ficheiros combinados com sucesso: {sucessos}")
    if erros > 0:
        print(f"Ficheiros ignorados (erros/falta de par): {erros}")
    print(f"\nOs seus dados limpos estão agora na pasta: {OUTPUT_ROOT}")

if __name__ == "__main__":
    process_combined_parquets()