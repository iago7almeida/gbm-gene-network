import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def deploy_dashboard_to_neo4j(driver, json_path):
    print("📂 Lendo o arquivo JSON estrutural do Dashboard...")
    
    if not os.path.exists(json_path):
        print(f"❌ Erro: Arquivo não encontrado em {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as file:
        dashboard_data = json.load(file)

    title = dashboard_data.get("title", "Dashboard Oncológico Automático")
    version = dashboard_data.get("version", "2.4")
    content_str = json.dumps(dashboard_data)

    query = """
    MERGE (d:_Neodash_Dashboard {title: $title})
    SET d.version = $version,
        d.content = $content,
        d.user = $user,
        d.date = datetime()
    RETURN d.title AS title
    """

    print(f"🚀 Injetando o dashboard '{title}' no Neo4j Aura Workspace...")
    with driver.session() as session:
        result = session.run(query, title=title, version=version, content=content_str, user=USER)
        record = result.single()
        if record:
            print(f"✅ Dashboard '{record['title']}' instalado com sucesso e atribuído ao usuário '{USER}'!")

if __name__ == "__main__":
    print("🔌 Conectando ao Neo4j Aura...")
    config_file = "neo4j/neodash_config.json" 
    
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
        print("✅ Conexão estabelecida!\n")
        
        deploy_dashboard_to_neo4j(driver, config_file)

    except Exception as e:
        print(f"❌ Erro durante o deploy: {e}")
    finally:
        driver.close()