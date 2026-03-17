import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def create_constraints(driver):
    """
    Cria um índice de unicidade. Isso é CRÍTICO para a performance.
    Garante que o banco não crie genes duplicados e acelera muito a busca.
    """
    print("⚙️ Configurando o banco de dados...")
    query = "CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE"
    with driver.session() as session:
        session.run(query)
    print("✅ Constraint de unicidade criada.")

def load_nodes(driver, nodes_path):
    print("\n🧬 Iniciando o carregamento dos Nós (Genes)...")
    df_nodes = pd.read_csv(nodes_path)
    records = df_nodes.to_dict('records')
    
    query = """
    UNWIND $batch AS row
    MERGE (g:Gene {id: row.gene_id})
    """
    with driver.session() as session:
        session.run(query, batch=records)
    print(f"✅ {len(records)} genes carregados com sucesso.")

def load_edges(driver, edges_path, batch_size=10000):
    """
    Lê o CSV de arestas e envia em lotes para evitar estouro de memória na rede.
    """
    print("\n🕸️ Iniciando o carregamento das Arestas (Correlações)...")
    df_edges = pd.read_csv(edges_path)
    records = df_edges.to_dict('records')
    
    query = """
    UNWIND $batch AS row
    MATCH (source:Gene {id: row.gene_source})
    MATCH (target:Gene {id: row.gene_target})
    MERGE (source)-[r:CO_EXPRESSES_WITH]->(target)
    SET r.weight = toFloat(row.weight)
    """
    
    total_batches = (len(records) // batch_size) + 1
    
    with driver.session() as session:
        for i in range(total_batches):
            batch = records[i*batch_size : (i+1)*batch_size]
            if batch:
                session.run(query, batch=batch)
                print(f"   ⏳ Lote {i+1}/{total_batches} enviado...")
                
    print(f"✅ {len(records)} conexões mapeadas no banco com sucesso.")

if __name__ == "__main__":
    nodes_file = "data/processed/gbm_genes_nodes.csv"
    edges_file = "data/processed/gbm_network_edges.csv"
    
    if not os.path.exists(nodes_file) or not os.path.exists(edges_file):
        print("❌ Erro: Arquivos CSV não encontrados na pasta data/processed/")
        exit()

    print("🔌 Conectando ao Neo4j Aura...")
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
        print("✅ Conexão estabelecida!")
        
        # Execução do Pipeline
        create_constraints(driver)
        load_nodes(driver, nodes_file)
        load_edges(driver, edges_file, batch_size=15000)
        
    except Exception as e:
        print(f"❌ Falha ao conectar ou processar: {e}")
    finally:
        driver.close()
        print("\n🎉 Processo de ingestão finalizado!")