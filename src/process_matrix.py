import pandas as pd
import numpy as np
import os

def clean_expression_data(df, top_n_genes=2000):
    """
    Limpa a matriz bruta e seleciona apenas os Genes Altamente Variáveis (HVGs)
    para evitar explosão de memória RAM e focar no sinal biológico real.
    """
    print("🧹 A iniciar a limpeza dos dados e seleção de HVGs...")
    
    # 1. Remover linhas que são metadados do STAR-Counts
    genes_to_drop = [index for index in df.index if str(index).startswith('N_') or str(index).startswith('__')]
    df_clean = df.drop(index=genes_to_drop)
    
    # 2. Remover genes que têm expressão zero em todas as amostras
    df_clean = df_clean.loc[(df_clean != 0).any(axis=1)]
    
    # 3. O FILTRO SALVA-VIDAS: Calcular a variância de cada gene
    print(f"📉 Calculando variância para {df_clean.shape[0]} genes...")
    variances = df_clean.var(axis=1)
    
    # 4. Manter apenas os 'top_n_genes' com a maior variância
    # Isso reduz a matriz de ~60.000 linhas para apenas 2.000 (ou o valor que você definir)
    top_genes_index = variances.nlargest(top_n_genes).index
    df_clean = df_clean.loc[top_genes_index]
    
    print(f"✅ Limpeza concluída. Matriz reduzida para os {df_clean.shape[0]} genes mais variáveis.")
    return df_clean

def calculate_coexpression_network(df, correlation_threshold=0.8):
    """
    Calcula a matriz de correlação de Pearson e transforma-a numa lista de arestas para o Neo4j.
    """
    print("🧮 A calcular a matriz de Correlação de Pearson...")
    
    # A transposição (.T) é necessária porque o pandas correlaciona colunas. 
    corr_matrix = df.T.corr(method='pearson')
    
    print("🕸️ A transformar a matriz numa Lista de Arestas (Edge List)...")
    # Transforma a matriz quadrada num formato longo (Gene_A, Gene_B, Valor)
    edges = corr_matrix.stack().reset_index()
    edges.columns = ['gene_source', 'gene_target', 'weight']
    
    # 1. Remover auto-correlações (Gene A com Gene A = 1.0)
    edges = edges[edges['gene_source'] != edges['gene_target']]
    
    # 2. Filtrar apenas correlações fortes (positivas ou negativas)
    # Isso evita que o nosso banco de grafos fique sobrecarregado com conexões fracas e irrelevantes
    strong_edges = edges[
        (edges['weight'] >= correlation_threshold) | 
        (edges['weight'] <= -correlation_threshold)
    ]
    
    # 3. Remover duplicados (A->B é o mesmo que B->A na correlação de Pearson)
    # Ordenar os nomes dos genes para garantir que a dupla fica na mesma ordem e remover duplicados
    strong_edges['pair'] = strong_edges.apply(lambda x: tuple(sorted([x['gene_source'], x['gene_target']])), axis=1)
    strong_edges = strong_edges.drop_duplicates(subset=['pair']).drop(columns=['pair'])
    
    print(f"✅ Rede construída! Foram geradas {strong_edges.shape[0]} interações fortes.")
    return strong_edges

if __name__ == "__main__":
    # Caminhos dos ficheiros
    input_file = "data/raw/matriz_gbm_bruta.csv"
    output_nodes = "data/processed/gbm_genes_nodes.csv"
    output_edges = "data/processed/gbm_network_edges.csv"
    
    # Verifica se o ficheiro bruto existe
    if not os.path.exists(input_file):
        print(f"❌ Erro: O ficheiro {input_file} não foi encontrado.")
        exit()
        
    # Carregar dados
    print("📂 A carregar a matriz bruta...")
    df_raw = pd.read_csv(input_file, index_col=0)
    
    # Pipeline de Processamento
    df_cleaned = clean_expression_data(df_raw)
    
    # Calcular arestas (Neste exemplo, limitamos a correlações muito fortes > 0.85 para testar)
    df_edges = calculate_coexpression_network(df_cleaned, correlation_threshold=0.85)
    
    # Guardar resultados localmente
    os.makedirs(os.path.dirname(output_edges), exist_ok=True)
    df_edges.to_csv(output_edges, index=False)
    
    # Guardar a lista de nós únicos para criar os vértices no Neo4j
    unique_genes = pd.DataFrame({'gene_id': pd.concat([df_edges['gene_source'], df_edges['gene_target']]).unique()})
    unique_genes.to_csv(output_nodes, index=False)
    
    print(f"💾 Ficheiros guardados com sucesso na pasta 'data/processed/'.")