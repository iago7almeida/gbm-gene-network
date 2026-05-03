"""
gene_symbol_mapper.py — Mapeamento ENSG → Gene Symbol (HGNC)

Converte IDs Ensembl (ex: ENSG00000141510) para nomes de genes legíveis
(ex: TP53) usando a API mygene com cache local persistente.

Uso:
    from gene_symbol_mapper import GeneMapper
    mapper = GeneMapper()
    symbol = mapper.get_symbol("ENSG00000141510")  # → "TP53"
    df = mapper.enrich_dataframe(df, ensg_column="gene_id")
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    import mygene
    MYGENE_AVAILABLE = True
except ImportError:
    MYGENE_AVAILABLE = False
    print("⚠️  mygene não encontrado. Rode: pip install mygene")


CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "external", "ensg_to_symbol.csv"
)

# Genes de alta relevância clínica no GBM — fallback hardcoded
GBM_KNOWN_GENES = {
    "ENSG00000141510": "TP53",
    "ENSG00000146648": "EGFR",
    "ENSG00000171862": "PTEN",
    "ENSG00000138413": "IDH1",
    "ENSG00000182054": "IDH2",
    "ENSG00000204209": "DAXX",
    "ENSG00000164362": "TERT",
    "ENSG00000073282": "TP63",
    "ENSG00000197888": "UGT1A6",
    "ENSG00000170558": "CDH2",
    "ENSG00000111276": "CDKN1B",
    "ENSG00000147889": "CDKN2A",
    "ENSG00000148773": "MKI67",
    "ENSG00000100030": "MAPK1",
    "ENSG00000105329": "TGFB1",
    "ENSG00000073756": "PTGS2",
    "ENSG00000157764": "BRAF",
    "ENSG00000213281": "NRAS",
    "ENSG00000174775": "HRAS",
    "ENSG00000171791": "BCL2",
    "ENSG00000149311": "ATM",
    "ENSG00000012048": "BRCA1",
    "ENSG00000139618": "BRCA2",
    "ENSG00000121879": "PIK3CA",
    "ENSG00000117020": "AKT1",
    "ENSG00000198793": "MTOR",
    "ENSG00000105976": "MET",
    "ENSG00000066468": "FGFR2",
    "ENSG00000068078": "FGFR3",
    "ENSG00000162413": "MGMT",
    "ENSG00000118046": "STK11",
    "ENSG00000116062": "MSH6",
    "ENSG00000095002": "MSH2",
    "ENSG00000076242": "MLH1",
    "ENSG00000134086": "VHL",
    "ENSG00000185686": "PRAME",
    "ENSG00000196712": "NF1",
    "ENSG00000186868": "NF2",
    "ENSG00000111640": "GAPDH",
    "ENSG00000075624": "ACTB",
}


class GeneMapper:
    """
    Mapper de IDs Ensembl (ENSG) para Gene Symbols (HGNC).

    Usa mygene API com cache local em CSV para performance.
    Fallback para genes GBM-relevantes hardcoded quando a API falha.
    """

    def __init__(self, cache_path=CACHE_PATH):
        self.cache_path = cache_path
        self.mapping = {}
        self._load_cache()

    def _load_cache(self):
        """Carrega o cache local se existir."""
        if os.path.exists(self.cache_path):
            df_cache = pd.read_csv(self.cache_path)
            self.mapping = dict(zip(
                df_cache['ensembl_id'].astype(str),
                df_cache['symbol'].astype(str)
            ))
            print(f"📂 Cache carregado: {len(self.mapping)} mapeamentos gene→symbol")
        else:
            # Inicializa com genes conhecidos
            self.mapping = dict(GBM_KNOWN_GENES)
            print("⚠️  Sem cache local. Usando fallback de genes conhecidos do GBM.")

    def _save_cache(self):
        """Persiste o mapeamento atual no CSV."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df = pd.DataFrame([
            {"ensembl_id": k, "symbol": v}
            for k, v in self.mapping.items()
            if v and v != "Unknown"
        ])
        df.to_csv(self.cache_path, index=False)
        print(f"💾 Cache salvo: {len(df)} mapeamentos em {self.cache_path}")

    def _strip_version(self, ensg_id):
        """Remove versão do ID Ensembl: ENSG00000141510.5 → ENSG00000141510"""
        return str(ensg_id).split('.')[0].strip()

    def query_batch(self, ensg_ids, batch_size=500):
        """
        Consulta a API mygene em lotes para mapear IDs desconhecidos.

        Args:
            ensg_ids: Lista de ENSG IDs a resolver
            batch_size: Tamanho do lote para a API
        """
        if not MYGENE_AVAILABLE:
            print("❌ mygene não está instalado. Usando apenas fallback.")
            return

        # Filtrar apenas IDs não mapeados
        unknown_ids = [
            self._strip_version(eid)
            for eid in ensg_ids
            if self._strip_version(eid) not in self.mapping
        ]

        if not unknown_ids:
            print("✅ Todos os genes já estão no cache.")
            return

        unique_unknowns = list(set(unknown_ids))
        print(f"🔍 Consultando mygene API para {len(unique_unknowns)} genes desconhecidos...")

        mg = mygene.MyGeneInfo()
        n_batches = (len(unique_unknowns) + batch_size - 1) // batch_size

        resolved = 0
        for i in tqdm(range(n_batches), desc="Lotes API"):
            batch = unique_unknowns[i * batch_size:(i + 1) * batch_size]
            try:
                results = mg.querymany(
                    batch,
                    scopes='ensembl.gene',
                    fields='symbol',
                    species='human',
                    returnall=False,
                    as_dataframe=False
                )
                for r in results:
                    query_id = r.get('query', '')
                    symbol = r.get('symbol', None)
                    if symbol:
                        self.mapping[query_id] = symbol
                        resolved += 1
            except Exception as e:
                print(f"   ⚠️  Erro no lote {i+1}: {e}")

        print(f"✅ Resolvidos {resolved}/{len(unique_unknowns)} genes via API.")
        self._save_cache()

    def get_symbol(self, ensg_id):
        """
        Retorna o gene symbol para um ENSG ID.

        Returns:
            str: Gene symbol (ex: 'TP53') ou o ID original se não encontrado
        """
        clean_id = self._strip_version(ensg_id)
        return self.mapping.get(clean_id, clean_id)

    def enrich_dataframe(self, df, ensg_column="gene_id", symbol_column="gene_symbol"):
        """
        Adiciona uma coluna de gene symbols a um DataFrame.

        Args:
            df: DataFrame com IDs Ensembl
            ensg_column: Nome da coluna com os ENSG IDs
            symbol_column: Nome da coluna de output

        Returns:
            DataFrame com a coluna symbol_column adicionada
        """
        # Primeiro, resolve IDs desconhecidos via API
        if ensg_column in df.columns:
            ids_to_resolve = df[ensg_column].unique().tolist()
        else:
            # Assume que os IDs estão no index
            ids_to_resolve = df.index.tolist()

        self.query_batch(ids_to_resolve)

        # Aplica o mapeamento
        if ensg_column in df.columns:
            df[symbol_column] = df[ensg_column].apply(
                lambda x: self.get_symbol(x)
            )
        else:
            df[symbol_column] = [self.get_symbol(idx) for idx in df.index]

        n_resolved = (df[symbol_column] != df.get(ensg_column, df.index)).sum()
        print(f"🧬 {n_resolved}/{len(df)} genes mapeados para symbols.")
        return df

    def get_display_name(self, ensg_id):
        """
        Retorna nome para display: 'TP53 (ENSG00000141510)' ou 'ENSG...' se desconhecido.
        """
        clean_id = self._strip_version(ensg_id)
        symbol = self.mapping.get(clean_id)
        if symbol and symbol != clean_id:
            return f"{symbol} ({clean_id})"
        return clean_id

    def bulk_resolve(self, ensg_ids):
        """
        Resolve uma lista de IDs e retorna um dicionário {ensg: symbol}.
        """
        self.query_batch(ensg_ids)
        return {
            self._strip_version(eid): self.get_symbol(eid)
            for eid in ensg_ids
        }


if __name__ == "__main__":
    mapper = GeneMapper()

    # Teste com genes GBM bem conhecidos
    test_ids = [
        "ENSG00000141510",  # TP53
        "ENSG00000146648",  # EGFR
        "ENSG00000171862",  # PTEN
        "ENSG00000138413",  # IDH1
        "ENSG00000162413",  # MGMT
    ]

    print("\n🧪 Teste de Mapeamento:")
    for eid in test_ids:
        print(f"   {eid} → {mapper.get_symbol(eid)}")

    # Se mygene está disponível, tenta resolver os genes do projeto
    nodes_path = "data/processed/gbm_genes_nodes.csv"
    if os.path.exists(nodes_path):
        print(f"\n📂 Enriquecendo os genes do projeto ({nodes_path})...")
        df_nodes = pd.read_csv(nodes_path)
        # Limpar versão
        df_nodes['gene_id_clean'] = df_nodes['gene_id'].astype(str).str.split('.').str[0]
        df_nodes = mapper.enrich_dataframe(df_nodes, ensg_column='gene_id_clean')
        print(f"\n🏅 Top 20 Genes Mapeados:")
        mapped = df_nodes[df_nodes['gene_symbol'] != df_nodes['gene_id_clean']]
        print(mapped[['gene_id_clean', 'gene_symbol']].head(20).to_string(index=False))
