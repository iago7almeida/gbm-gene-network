import requests
import json
import pandas as pd
import tarfile
import io
import os

def query_gdc_api(project_id="TCGA-GBM", max_files=200):
    """
    Consulta a API do GDC para encontrar arquivos de expressão gênica.
    Limitado a 'max_files' para testes iniciais de pipeline.
    """
    files_endpoint = "https://api.gdc.cancer.gov/files"
    
    # Filtros para pegar apenas Quantificação de Expressão (RNA-Seq)
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "files.data_category", "value": ["Transcriptome Profiling"]}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}}
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id",
        "format": "JSON",
        "size": max_files # Comece pequeno para validar o código
    }

    print("🔍 Consultando a API por metadados...")
    response = requests.get(files_endpoint, params=params)
    response.raise_for_status()
    
    file_metadata = response.json()["data"]["hits"]
    file_ids = [f["file_id"] for f in file_metadata]
    
    print(f"✅ Encontrados {len(file_ids)} arquivos para download.")
    return file_ids, file_metadata

def download_and_extract_matrix(file_ids, metadata):
    """
    Baixa os arquivos via API e constrói a matriz no Pandas.
    Otimizado com pd.concat e proteção contra amostras duplicadas do mesmo paciente.
    """
    data_endpoint = "https://api.gdc.cancer.gov/data"
    
    print("☁️ Iniciando o download dos dados de expressão...")
    response = requests.post(data_endpoint, data=json.dumps({"ids": file_ids}), headers={"Content-Type": "application/json"})
    response.raise_for_status()

    tar = tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz")
    
    # Usaremos uma lista para guardar as colunas (muito mais rápido que .join num loop)
    dfs = []

    print("🧩 Montando a matriz de expressão...")
    for member in tar.getmembers():
        if member.name.endswith('.tsv'):
            f = tar.extractfile(member)
            content = f.read()
            
            # 1. Busca Dinâmica de Cabeçalho
            lines = content.split(b'\n')
            header_idx = 0
            for i, line in enumerate(lines):
                if b'gene_id' in line:
                    header_idx = i
                    break
            
            # 2. Carrega o Pandas
            valid_content = b'\n'.join(lines[header_idx:])
            df_temp = pd.read_csv(io.BytesIO(valid_content), sep='\t')
            
            df_temp.columns = df_temp.columns.str.strip()
            df_temp = df_temp[['gene_id', 'unstranded']]
            df_temp.set_index('gene_id', inplace=True)
            
            # 3. Mapeia o ID do paciente e o ID do arquivo
            file_name = os.path.basename(member.name)
            submitter_id = file_name # Fallback
            file_uuid = "unknown"
            
            for item in metadata:
                if item["file_name"] == file_name:
                    submitter_id = item['cases'][0]['submitter_id']
                    file_uuid = item['file_id']
                    break
            
            # 4. Cria um nome ÚNICO: ID_Paciente + Prefixo do UUID do arquivo
            col_name = f"{submitter_id}_{file_uuid[:8]}"
            
            df_temp.rename(columns={'unstranded': col_name}, inplace=True)
            dfs.append(df_temp)

    # 5. Junta todas as amostras na matriz final de uma única vez
    print("🚀 Concatenando todas as amostras...")
    if dfs:
        matriz_expressao = pd.concat(dfs, axis=1)
    else:
        matriz_expressao = pd.DataFrame()

    return matriz_expressao

# --- Execução Principal ---
if __name__ == "__main__":
    # 1. Pegar os IDs dos arquivos (Testando com 5 amostras primeiro)
    ids, meta = query_gdc_api(max_files=200)
    
    # 2. Baixar e montar o DataFrame
    df_matriz = download_and_extract_matrix(ids, meta)
    
    print("\n📊 Matriz Final:")
    print(df_matriz.head(20))
    
    # 3. Salvar o modelo localmente
    output_path = "data/raw/matriz_gbm_bruta.csv"
    df_matriz.to_csv(output_path, index=True, index_label='gene_id')
    print(f"\n💾 Arquivo salvo com sucesso em: {output_path}")