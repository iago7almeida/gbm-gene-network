# 🧠 Mapeamento de Redes de Interação Gênica no Glioblastoma (TCGA)

## 📌 Sobre o Projeto
Este projeto de Bioinformática e Ciência de Dados visa construir um modelo de redes complexas para mapear a co-expressão génica no Glioblastoma Multiforme (GBM). O objetivo é identificar "nós" centrais (hubs) na rede de regulação do tumor utilizando dados de sequenciação de RNA (RNA-Seq).

Os dados brutos são extraídos via API diretamente do **GDC (Genomic Data Commons)**, referentes ao projeto `TCGA-GBM`.

## 🏗️ Arquitetura e Tecnologias
O pipeline analítico foi desenhado com foco em automação, escalabilidade e análise de sistemas complexos:
* **Extração e Processamento (Python):** Utilização de `requests` para interagir com a API do GDC e `pandas`/`scipy` para limpeza e cálculo da matriz de Correlação de Pearson.
* **Modelagem de Grafos (Neo4j):** O modelo relacional de genes é armazenado num banco de dados orientado a grafos. Algoritmos de centralidade são aplicados para descobrir padrões biológicos. A visualização interativa é suportada via **NeoDash**.
* **Orquestração na Nuvem (AWS):** O fluxo de extração e processamento é preparado para execução via AWS CodeBuild, com armazenamento de dados limpos e matrizes finais em buckets do Amazon S3.

## 🚀 Configuração Inicial (Setup)

### 1. Clonar o Repositório e Instalar Dependências
```bash
git clone [https://github.com/teu-usuario/gbm-gene-network.git](https://github.com/teu-usuario/gbm-gene-network.git)
cd gbm-gene-network
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt


## 📁 Estrutura do Repositório

```text
gbm-gene-network/
│
├── data/
│   ├── raw/                 # Dados brutos baixados da API (.tsv.gz)
│   ├── processed/           # Lista de nós e arestas filtradas (.csv)
│   └── reports/             # Relatórios tabulares extraídos do Neo4j
│
├── neo4j/
│   └── neodash_config.json  # Estrutura do painel visual exportada
│
├── src/
│   ├── extract_gdc.py       # Script de extração da API
│   ├── process_matrix.py    # Cálculo de HVGs e matriz de correlação
│   ├── load_graph.py        # Ingestão em lotes para o Neo4j
│   ├── analyze_network.py   # Algoritmos Cypher rodando via Python
│   └── deploy_dashboard.py  # Deploy programático do NeoDash no Aura
│
├── .env.example             # Template para variáveis de ambiente
├── requirements.txt         # Dependências do projeto
└── README.md ```