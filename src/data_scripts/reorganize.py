import shutil
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURAÇÕES
# ============================================================
# A pasta de onde vamos ler os ficheiros misturados
SOURCE_ROOT = Path("IPqM-Fall/OLD/trials_90hz_combined")

# A nova pasta raiz onde a hierarquia será construída
DEST_ROOT = Path("IPqM-Fall/dataset_hierarquico")

def reorganize_files():
    # Verifica se a pasta de origem existe
    if not SOURCE_ROOT.exists():
        print(f"Erro: A pasta {SOURCE_ROOT} não foi encontrada.")
        return

    # Lista todos os ficheiros parquet
    files = list(SOURCE_ROOT.glob("*.parquet"))
    
    if not files:
        print(f"Nenhum ficheiro encontrado em {SOURCE_ROOT}.")
        return

    print(f"A iniciar reorganização de {len(files)} ficheiros...\n")
    
    sucessos = 0
    erros = 0

    for file_path in tqdm(files, desc="A organizar pastas"):
        try:
            # O nome do ficheiro atual. Ex: "ID1_CHEST_ADL_1_trial1_combined.parquet"
            filename = file_path.name
            
            # Separar pelas underlines
            parts = filename.split('_')
            
            # Extrair os blocos de informação
            subject_id = parts[0]   # Ex: "ID1"
            sensor_pos = parts[1]   # Ex: "CHEST"
            
            # Reconstruir o nome do ficheiro sem o ID e o Sensor (para ficar mais limpo)
            # parts[2:] pega em tudo a partir de "ADL_1..." ou "OM_3..."
            new_filename = "_".join(parts[2:]) 
            
            # Criar o caminho de destino hierárquico
            # Ex: IPqM-Fall/dataset_hierarquico/ID1/CHEST/
            dest_dir = DEST_ROOT / subject_id / sensor_pos
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Caminho final do ficheiro
            dest_path = dest_dir / new_filename
            
            # Copiar o ficheiro (usamos copy2 para preservar os metadados de criação/modificação)
            # Se preferir MOVER (para poupar espaço no disco), mude de shutil.copy2 para shutil.move
            shutil.copy2(file_path, dest_path)
            
            sucessos += 1
            
        except Exception as e:
            print(f"\nErro ao processar {filename}: {e}")
            erros += 1

    print("\n" + "="*50)
    print("REORGANIZAÇÃO CONCLUÍDA!")
    print("="*50)
    print(f"Ficheiros organizados com sucesso: {sucessos}")
    if erros > 0:
        print(f"Erros encontrados: {erros}")
    print(f"\nA sua nova estrutura de pastas encontra-se em: {DEST_ROOT}")
    print("Exemplo de caminho: IPqM-Fall/dataset_hierarquico/ID1/CHEST/ADL_1_trial1_combined.parquet")

if __name__ == "__main__":
    reorganize_files()