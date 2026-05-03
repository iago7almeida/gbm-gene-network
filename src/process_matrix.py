import pandas as pd
import numpy as np
import os

def normalize_log1p(df):
    """
    Aplica normalização log1p (log(1 + count)) à matriz de expressão.

    CORREÇÃO CRÍTICA: RNA-Seq raw counts NÃO podem ser usados diretamente para
    cálculo de variância e correlação. Genes com alta expressão absoluta dominam
    artificialmente as métricas, independente do sinal biológico real.

    log1p é o padrão da área para:
    - Estabilizar a variância entre genes de baixa e alta expressão
    - Aproximar a distribuição de counts de uma distribuição normal
    - Tornar a correlação de Pearson/Spearman matematicamente válida
    """
    print("🔢 Aplicando normalização log1p (log(1 + count))...")
    df_log = np.log1p(df.astype(float))
    print("✅ Normalização concluída.")
    return df_log


def clean_expression_data(df, top_n_genes=3000):
    """
    Limpa a matriz bruta, normaliza e seleciona os Genes Altamente Variáveis (HVGs).

    CORREÇÃO: top_n_genes aumentado de 2000 → 3000 para aproveitar o dataset maior
    (~600 amostras vs 200 anteriores) e capturar mais sinal biológico.
    """
    print("🧹 Iniciando a limpeza dos dados e seleção de HVGs...")

    # 1. Remover linhas que são metadados do STAR-Counts
    genes_to_drop = [
        index for index in df.index
        if str(index).startswith('N_') or str(index).startswith('__')
    ]
    df_clean = df.drop(index=genes_to_drop)
    print(f"   🗑️  Removidas {len(genes_to_drop)} linhas de metadados STAR.")

    # 2. Remover genes com expressão zero em todas as amostras
    mask_nonzero = (df_clean != 0).any(axis=1)
    df_clean = df_clean.loc[mask_nonzero]
    print(f"   🗑️  Removidos genes com zero em todas as amostras. Restam: {df_clean.shape[0]}")

    # 3. CORREÇÃO: Normalizar ANTES de calcular variância
    df_log = normalize_log1p(df_clean)

    # 4. Calcular variância nos dados normalizados
    print(f"📉 Calculando variância para {df_log.shape[0]} genes (dados normalizados)...")
    variances = df_log.var(axis=1)

    # 5. Manter apenas os 'top_n_genes' com maior variância
    top_genes_index = variances.nlargest(top_n_genes).index
    df_hvg = df_log.loc[top_genes_index]

    print(f"✅ Limpeza concluída. Matriz reduzida para os {df_hvg.shape[0]} genes mais variáveis.")
    return df_hvg


def calculate_coexpression_network(df, correlation_threshold=0.8, method='spearman'):
    """
    Calcula a matriz de correlação e transforma-a numa lista de arestas para o Neo4j.

    CORREÇÃO: Substituída correlação de Pearson por Spearman.
    - Pearson assume distribuição normal e é sensível a outliers.
    - Spearman é baseado em ranks, mais robusto para dados biológicos com distribuição
      assimétrica residual mesmo após normalização log1p.
    - Mesmo com log1p, a correlação de Spearman é o padrão de facto para RNA-Seq.
    """
    print(f"🧮 Calculando matriz de Correlação de {method.capitalize()}...")
    print(f"   (Matriz: {df.shape[0]} genes × {df.shape[1]} amostras — pode demorar alguns minutos)")

    # A transposição (.T) é necessária porque o pandas correlaciona colunas
    corr_matrix = df.T.corr(method=method)

    # Renomear eixos para evitar conflito de 'gene_id' no reset_index()
    corr_matrix.index.name = 'source_idx'
    corr_matrix.columns.name = 'target_idx'

    print("🕸️  Transformando a matriz numa Lista de Arestas (Edge List)...")
    edges = corr_matrix.stack().reset_index()
    edges.columns = ['gene_source', 'gene_target', 'weight']

    # 1. Remover auto-correlações (Gene A ↔ Gene A = 1.0)
    edges = edges[edges['gene_source'] != edges['gene_target']]

    # 2. Filtrar correlações fortes (acima do threshold, positivas ou negativas)
    strong_edges = edges[
        (edges['weight'] >= correlation_threshold) |
        (edges['weight'] <= -correlation_threshold)
    ].copy()

    # 3. Remover duplicados simétricos (A→B é o mesmo que B→A em correlação não-dirigida)
    strong_edges['pair'] = strong_edges.apply(
        lambda x: tuple(sorted([x['gene_source'], x['gene_target']])), axis=1
    )
    strong_edges = strong_edges.drop_duplicates(subset=['pair']).drop(columns=['pair'])

    print(f"✅ Rede construída! {strong_edges.shape[0]} interações fortes "
          f"(threshold={correlation_threshold}, método={method}).")
    return strong_edges


if __name__ == "__main__":
    input_file = "data/raw/matriz_gbm_bruta.csv"
    output_nodes = "data/processed/gbm_genes_nodes.csv"
    output_edges = "data/processed/gbm_network_edges.csv"

    if not os.path.exists(input_file):
        print(f"❌ Erro: O ficheiro {input_file} não foi encontrado.")
        exit()

    print("📂 Carregando a matriz bruta...")
    df_raw = pd.read_csv(input_file, index_col=0)
    print(f"   Shape bruta: {df_raw.shape[0]} genes × {df_raw.shape[1]} amostras")

    # Pipeline de Processamento com normalização correta
    df_cleaned = clean_expression_data(df_raw, top_n_genes=3000)

    # Calcular arestas com Spearman
    df_edges = calculate_coexpression_network(df_cleaned, correlation_threshold=0.8, method='spearman')

    # Guardar resultados
    os.makedirs(os.path.dirname(output_edges), exist_ok=True)
    df_edges.to_csv(output_edges, index=False)

    unique_genes = pd.DataFrame({
        'gene_id': pd.concat([df_edges['gene_source'], df_edges['gene_target']]).unique()
    })
    unique_genes.to_csv(output_nodes, index=False)

    print(f"\n💾 Arquivos guardados com sucesso em 'data/processed/'.")
    print(f"   Nós (genes) : {unique_genes.shape[0]}")
    print(f"   Arestas      : {df_edges.shape[0]}")