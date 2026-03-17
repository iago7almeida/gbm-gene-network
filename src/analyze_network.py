import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def run_query_to_df(driver, query, query_name):
    """Executa a query Cypher e retorna os resultados num DataFrame Pandas"""
    print(f"🔍 Executando análise: {query_name}...")
    with driver.session() as session:
        result = session.run(query)
        # Converte o resultado do Neo4j para uma lista de dicionários, depois para Pandas
        records = [record.data() for record in result]
        df = pd.DataFrame(records)
        return df

if __name__ == "__main__":
    print("🔌 Conectando ao Neo4j Aura para Análise...")
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
        print("✅ Conexão estabelecida!\n")
        
        # 1. Query: Top Hubs (Degree Centrality)
        query_hubs = """
        MATCH (g:Gene)-[r:CO_EXPRESSES_WITH]-()
        RETURN g.id AS Gene, count(r) AS Total_Conexoes
        ORDER BY Total_Conexoes DESC
        LIMIT 10
        """
        df_hubs = run_query_to_df(driver, query_hubs, "Top 10 Hubs")
        
        # 2. Query: Master Regulators (Alcance Global)
        query_master = """
        MATCH (alvo:Gene)-[:CO_EXPRESSES_WITH]-(vizinho:Gene)-[:CO_EXPRESSES_WITH]-(secundario:Gene)
        WHERE alvo <> secundario
        RETURN alvo.id AS Gene, count(DISTINCT secundario) AS Alcance_Sistemico
        ORDER BY Alcance_Sistemico DESC
        LIMIT 10
        """
        df_master = run_query_to_df(driver, query_master, "Master Regulators")
        
        # Exibindo os resultados no terminal
        print("\n🏆 RESULTADOS DA ANÁLISE:")
        print("-" * 30)
        print("TOP 10 HUBS (Conexões Diretas):")
        print(df_hubs)
        print("\nMASTER REGULATORS (Alcance Global):")
        print(df_master)
        
        # Salvando os relatórios localmente
        os.makedirs("../data/reports", exist_ok=True)
        df_hubs.to_csv("../data/reports/top_hubs.csv", index=False)
        df_master.to_csv("../data/reports/master_regulators.csv", index=False)
        print("\n💾 Relatórios salvos com sucesso na pasta 'data/reports/'")

    except Exception as e:
        print(f"❌ Erro durante a análise: {e}")
    finally:
        driver.close()