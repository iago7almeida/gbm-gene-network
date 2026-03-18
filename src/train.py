import os
import pandas as pd
import joblib
import boto3
from botocore.exceptions import NoCredentialsError
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

def load_and_merge_data(clinical_path, expression_path, nodes_path):
    print("📂 Carregando gabarito clínico...")
    df_clinical = pd.read_csv(clinical_path)
    df_clinical['paciente_id'] = df_clinical['paciente_id'].astype(str).str.strip()
    
    print("🧠 Lendo a lista de Genes de Alta Confiança (Neo4j)...")
    df_nodes = pd.read_csv(nodes_path)
    
    if 'gene_id' in df_nodes.columns:
        genes_validos = set(df_nodes['gene_id'].astype(str).str.split('.').str[0].str.strip())
    else:
        genes_validos = set(df_nodes.iloc[:, 0].astype(str).str.split('.').str[0].str.strip())
        
    print(f"🎯 Procurando por {len(genes_validos)} genes VIPs na matriz...")
    
    print("⏳ Lendo a matriz bruta em blocos (Chunks) para salvar RAM...")
    chunks = []
    
    for chunk in pd.read_csv(expression_path, index_col=0, chunksize=5000):
        # Corta a versão (.14) e espaços do índice da matriz
        chunk.index = chunk.index.astype(str).str.split('.').str[0].str.strip()
        
        # Filtra mantendo apenas os VIPs
        chunk_filtrado = chunk[chunk.index.isin(genes_validos)].copy()
        
        if not chunk_filtrado.empty:
            # Remove duplicatas acidentais
            chunk_filtrado = chunk_filtrado[~chunk_filtrado.index.duplicated(keep='first')]
            chunks.append(chunk_filtrado)
            
    if not chunks:
        print("❌ ERRO GRAVE: Nenhum gene da lista do Neo4j foi encontrado na matriz bruta.")
        return pd.DataFrame()
        
    df_expr = pd.concat(chunks)
    print(f"🧬 SUCESSO! Pescamos {df_expr.shape[0]} genes cruciais da matriz gigante.")
    
    print("🔄 Transpondo a matriz enxuta...")
    df_expr = df_expr.T 
    df_expr.columns = df_expr.columns.astype(str)
    df_expr.index = df_expr.index.astype(str).str.strip().str[:12]
    
    print("🔗 Cruzando dados clínicos com a expressão gênica...")
    df_merged = pd.merge(df_clinical, df_expr, left_on='paciente_id', right_index=True, how='inner')
    
    if df_merged.empty:
        print("❌ ALERTA: A fusão resultou em 0 pacientes.")
    else:
        print(f"✅ Fusão de sucesso com {df_merged.shape[0]} pacientes e {df_expr.shape[1]} genes ultrasselecionados!")
        
    return df_merged

def preprocess_data(df):
    print("⚙️ Pré-processando variáveis categóricas e explodindo tratamentos...")
    
    y = df['classe_risco']
    colunas_ignoradas = ['paciente_id', 'status_vital', 'dias_sobrevivencia', 'classe_risco']
    X = df.drop(columns=[col for col in colunas_ignoradas if col in df.columns])
    
    if 'tipos_tratamento' in X.columns:
        print("💊 Separando múltiplos tratamentos em colunas independentes (One-Hot Encoding)...")
        tratamentos_binarios = X['tipos_tratamento'].str.get_dummies(sep=' | ')
        
        X = pd.concat([X, tratamentos_binarios], axis=1)
        X = X.drop(columns=['tipos_tratamento'])
    
    # Converte as variáveis de texto simples (Gênero, Raça) em números
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col].astype(str))
        
    X = X.fillna(0)
    X.columns = X.columns.astype(str)
    
    return X, y

def train_and_evaluate(X, y):
    print("🧠 Treinando o algoritmo Random Forest com calibração anti-ruído...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # ⚙️ CONFIGURAÇÃO SÊNIOR: 'Podando' a árvore para ela ignorar o ruído genético
    rf_model = RandomForestClassifier(
        n_estimators=300,           # Mais árvores para diluir o erro
        max_depth=5,                # Árvores mais rasas (antes era 10, ele estava decorando muito)
        min_samples_split=10,       # Exige pelo menos 10 pacientes para criar uma nova regra
        min_samples_leaf=5,         # Cada conclusão final precisa ter no mínimo 5 pacientes
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    accuracy = rf_model.score(X_test, y_test)
    print(f"🎯 Acurácia do Modelo: {accuracy * 100:.2f}%")
    
    # Extrai os fatores mais importantes
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importancia': rf_model.feature_importances_
    }).sort_values(by='Importancia', ascending=False)
    
    print("\n🏆 Top 15 Fatores Mais Críticos (Clínica + Genética):")
    print(importances.head(15).to_string(index=False))
    
    return rf_model

def upload_to_s3(file_name, bucket_name):
    print(f"☁️ Iniciando upload de {file_name} para o S3 (Bucket: {bucket_name})...")
    try:
        # Na nuvem, o boto3 herda automaticamente as permissões da Role do CodeBuild!
        s3_client = boto3.client('s3')
        s3_client.upload_file(file_name, bucket_name, os.path.basename(file_name))
        print("✅ Upload concluído com sucesso!")
    except Exception as e:
        print(f"❌ Erro durante o upload para o S3: {e}")

if __name__ == "__main__":
    # Caminhos dos arquivos (Ajuste o caminho da expressão conforme o arquivo limpo gerado no Colab)
    CLINICAL_PATH = "data/processed/clinical_gbm_completo.csv"
    EXPRESSION_PATH = "data/raw/matriz_gbm_bruta.csv" 
    MODEL_PATH = "models/modelo_sobrevivencia_gbm_v1.pkl"
    BUCKET_NAME = "gbm-ml-models-bucket"
    NODES_PATH = "data/processed/gbm_genes_nodes.csv"
    
    # 1. Carrega e cruza os dados
    df_completo = load_and_merge_data(CLINICAL_PATH, EXPRESSION_PATH, NODES_PATH)
    
    # TRAVA DE SEGURANÇA: Evita que o script quebre se a matriz voltar vazia
    if df_completo.empty:
        print("\n⚠️ Pipeline interrompido: Não há dados suficientes para treinar o modelo. Verifique os arquivos CSV.")
        exit()
    
    # 2. Prepara os dados para a máquina
    X, y = preprocess_data(df_completo)
    
    # 3. Treina o modelo
    modelo = train_and_evaluate(X, y)
    
    # 4. Salva e envia para a AWS
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    import joblib # Adicione import joblib lá no topo do arquivo se não estiver
    joblib.dump(modelo, MODEL_PATH)
    print(f"\n💾 Modelo salvo localmente em: {MODEL_PATH}")
    
    upload_to_s3(MODEL_PATH, BUCKET_NAME)