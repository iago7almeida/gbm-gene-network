import requests
import json
import pandas as pd
import tarfile
import io
import os
import time

# Tamanho máximo de IDs por requisição POST para evitar timeout/erro 413 da API do GDC
GDC_DOWNLOAD_BATCH_SIZE = 100

def query_gdc_api(project_id="TCGA-GBM", max_files=600):
    """
    Consulta a API do GDC para encontrar arquivos de expressão gênica.

    CORREÇÃO: max_files agora é 600 (total disponível no TCGA-GBM) em vez de 200.
    O valor 200 era um placeholder de teste que foi para produção, reduzindo o
    dataset efetivo em ~67% e prejudicando diretamente a qualidade do modelo.
    """
    files_endpoint = "https://api.gdc.cancer.gov/files"

    # Filtros para pegar apenas Quantificação de Expressão (RNA-Seq STAR-Counts)
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
        "size": max_files
    }

    print(f"🔍 Consultando a API por metadados (máx. {max_files} arquivos)...")
    response = requests.get(files_endpoint, params=params)
    response.raise_for_status()

    file_metadata = response.json()["data"]["hits"]
    file_ids = [f["file_id"] for f in file_metadata]

    print(f"✅ Encontrados {len(file_ids)} arquivos para download.")

    # CORREÇÃO: Pré-indexar metadados por file_name (O(1)) em vez de buscar em loop O(n²)
    metadata_index = {}
    for item in file_metadata:
        metadata_index[item["file_name"]] = item

    return file_ids, metadata_index


def download_batch(batch_ids):
    """
    Baixa um lote de arquivos via API do GDC e retorna o objeto tarfile em memória.
    Inclui retry automático com backoff exponencial para tolerância a falhas de rede.
    """
    data_endpoint = "https://api.gdc.cancer.gov/data"
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = requests.post(
                data_endpoint,
                data=json.dumps({"ids": batch_ids}),
                headers={"Content-Type": "application/json"},
                timeout=300
            )
            response.raise_for_status()
            return tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz")
        except requests.exceptions.RequestException as e:
            wait_secs = 2 ** attempt
            print(f"   ⚠️  Tentativa {attempt+1}/{max_retries} falhou: {e}. Aguardando {wait_secs}s...")
            time.sleep(wait_secs)

    raise RuntimeError(f"❌ Falha após {max_retries} tentativas de download.")


def parse_tsv_member(tar, member, metadata_index):
    """
    Extrai e parseia um único arquivo TSV do tarball.
    Retorna (col_name, df_temp) ou None se o arquivo for inválido.
    """
    f = tar.extractfile(member)
    if f is None:
        return None

    content = f.read()

    # Busca Dinâmica de Cabeçalho (linhas de metadados do STAR começam antes de gene_id)
    lines = content.split(b'\n')
    header_idx = 0
    for i, line in enumerate(lines):
        if b'gene_id' in line:
            header_idx = i
            break

    valid_content = b'\n'.join(lines[header_idx:])
    df_temp = pd.read_csv(io.BytesIO(valid_content), sep='\t')

    df_temp.columns = df_temp.columns.str.strip()
    if 'gene_id' not in df_temp.columns or 'unstranded' not in df_temp.columns:
        return None

    df_temp = df_temp[['gene_id', 'unstranded']]
    df_temp.set_index('gene_id', inplace=True)

    # CORREÇÃO: Lookup O(1) via dict pré-indexado
    file_name = os.path.basename(member.name)
    meta_item = metadata_index.get(file_name)

    if meta_item:
        submitter_id = meta_item['cases'][0]['submitter_id']
        file_uuid = meta_item['file_id']
    else:
        submitter_id = file_name
        file_uuid = "unknown00"

    col_name = f"{submitter_id}_{file_uuid[:8]}"
    df_temp.rename(columns={'unstranded': col_name}, inplace=True)
    return df_temp


def download_and_extract_matrix(file_ids, metadata_index):
    """
    Baixa os arquivos em lotes (batches) e constrói a matriz de expressão no Pandas.

    CORREÇÃO: Download em lotes de GDC_DOWNLOAD_BATCH_SIZE IDs por vez.
    Enviar >300 IDs em uma única requisição POST pode causar erro 413 (payload too large)
    ou timeout no servidor do GDC.
    """
    total = len(file_ids)
    n_batches = (total + GDC_DOWNLOAD_BATCH_SIZE - 1) // GDC_DOWNLOAD_BATCH_SIZE
    dfs = []

    print(f"☁️  Iniciando download em {n_batches} lotes de {GDC_DOWNLOAD_BATCH_SIZE} arquivos...")

    for batch_idx in range(n_batches):
        start = batch_idx * GDC_DOWNLOAD_BATCH_SIZE
        end = min(start + GDC_DOWNLOAD_BATCH_SIZE, total)
        batch_ids = file_ids[start:end]

        print(f"\n📦 Lote {batch_idx + 1}/{n_batches} ({len(batch_ids)} arquivos)...")
        tar = download_batch(batch_ids)

        for member in tar.getmembers():
            if member.name.endswith('.tsv'):
                result = parse_tsv_member(tar, member, metadata_index)
                if result is not None:
                    dfs.append(result)

    print(f"\n🚀 Concatenando {len(dfs)} amostras na matriz final...")
    if dfs:
        matriz_expressao = pd.concat(dfs, axis=1)
    else:
        matriz_expressao = pd.DataFrame()

    print(f"✅ Matriz construída: {matriz_expressao.shape[0]} genes × {matriz_expressao.shape[1]} amostras")
    return matriz_expressao


# --- Execução Principal ---
if __name__ == "__main__":
    # CORREÇÃO: 600 arquivos (cobertura total do TCGA-GBM) em vez de 200
    ids, meta_index = query_gdc_api(max_files=600)

    df_matriz = download_and_extract_matrix(ids, meta_index)

    print("\n📊 Matriz Final (primeiras linhas):")
    print(df_matriz.head(5))

    output_path = "data/raw/matriz_gbm_bruta.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_matriz.to_csv(output_path, index=True, index_label='gene_id')
    print(f"\n💾 Arquivo salvo com sucesso em: {output_path}")