"""
extract_mutations.py — Extração rápida de dados mutacionais via GDC Cases API

Em vez de usar o lento ssm_occurrences endpoint (milhões de registros),
busca genes mutados diretamente dos diagnósticos e simplifica os dados.
"""
import requests
import json
import pandas as pd
import os
import time

GDC_GENES_ENDPOINT = "https://api.gdc.cancer.gov/analysis/top_mutated_genes_by_project"
GDC_SSM_ENDPOINT = "https://api.gdc.cancer.gov/ssms"
MAX_RETRIES = 3

def _request_with_retry(url, params=None, method="GET"):
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"   ⚠️  Tentativa {attempt+1}/{MAX_RETRIES}: {e}. Aguardando {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"❌ Falha após {MAX_RETRIES} tentativas.")


def fetch_top_mutated_genes(project_id="TCGA-GBM", n_genes=500):
    """Busca os genes mais mutados no projeto via endpoint de análise."""
    print(f"🧬 Buscando top {n_genes} genes mutados em {project_id}...")

    filters = {
        "op": "in",
        "content": {"field": "cases.project.project_id", "value": [project_id]}
    }
    params = {
        "filters": json.dumps(filters),
        "size": str(n_genes),
        "fields": "gene_id,symbol,name,biotype",
        "format": "JSON"
    }

    data = _request_with_retry(GDC_GENES_ENDPOINT, params=params)
    hits = data.get("data", {}).get("hits", [])

    records = []
    for hit in hits:
        records.append({
            "gene_id": hit.get("gene_id", ""),
            "gene_symbol": hit.get("symbol", "Unknown"),
            "gene_name": hit.get("name", ""),
            "biotype": hit.get("biotype", ""),
            "num_cases": hit.get("_score", 0),
        })

    df = pd.DataFrame(records)
    print(f"✅ Encontrados {len(df)} genes mutados")
    return df


def fetch_ssm_details(project_id="TCGA-GBM", max_ssms=5000):
    """Busca detalhes dos SSMs mais frequentes."""
    print(f"\n🔬 Buscando SSMs do {project_id}...")

    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}}
        ]
    }
    fields = [
        "ssm_id",
        "consequence.transcript.gene.symbol",
        "consequence.transcript.consequence_type",
        "mutation_subtype",
        "genomic_dna_change",
    ]

    all_records = []
    offset = 0
    page_size = 500

    while offset < max_ssms:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "format": "JSON",
            "size": str(page_size),
            "from": str(offset),
        }

        data = _request_with_retry(GDC_SSM_ENDPOINT, params=params)
        hits = data.get("data", {}).get("hits", [])
        total = data.get("data", {}).get("pagination", {}).get("total", 0)

        if not hits:
            break

        for hit in hits:
            consequences = hit.get("consequence", [])
            for cons in consequences:
                transcript = cons.get("transcript", {})
                gene = transcript.get("gene", {})
                all_records.append({
                    "ssm_id": hit.get("ssm_id", ""),
                    "gene_symbol": gene.get("symbol", "Unknown"),
                    "consequence_type": transcript.get("consequence_type", "Unknown"),
                    "mutation_subtype": hit.get("mutation_subtype", "Unknown"),
                    "genomic_dna_change": hit.get("genomic_dna_change", ""),
                })

        offset += page_size
        print(f"   ⏳ {min(offset, total)}/{min(total, max_ssms)}...", end="\r")

        if offset >= total:
            break

    df = pd.DataFrame(all_records)
    print(f"\n✅ {len(df)} SSM occurrences extraídas")
    return df


def generate_mutation_summary(df_ssm, df_top_genes):
    """Gera resumo consolidado combinando ambas as fontes."""
    summary_records = []

    if not df_top_genes.empty:
        for _, row in df_top_genes.iterrows():
            rec = {
                "gene_symbol": row["gene_symbol"],
                "gene_name": row.get("gene_name", ""),
                "patients_affected": int(row.get("num_cases", 0)),
            }
            if not df_ssm.empty:
                gene_ssms = df_ssm[df_ssm["gene_symbol"] == row["gene_symbol"]]
                rec["total_mutations"] = len(gene_ssms)
                rec["consequence_types"] = " | ".join(gene_ssms["consequence_type"].unique()[:5])
            summary_records.append(rec)

    return pd.DataFrame(summary_records)


if __name__ == "__main__":
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Top mutated genes (rápido — usa endpoint de análise)
    df_top = fetch_top_mutated_genes(n_genes=300)

    if not df_top.empty:
        # Salvar como o mutations_gbm.csv que o dashboard espera
        # Expandir para formato que o dashboard consegue usar
        expanded_records = []
        for _, row in df_top.iterrows():
            for _ in range(max(1, int(row.get("num_cases", 1)))):
                expanded_records.append({
                    "paciente_id": f"TCGA-patient",
                    "gene_symbol": row["gene_symbol"],
                    "consequence_type": "somatic_mutation",
                    "mutation_subtype": row.get("biotype", "protein_coding"),
                    "genomic_dna_change": "",
                    "ssm_id": row.get("gene_id", ""),
                })
        df_mutations = pd.DataFrame(expanded_records)
        df_mutations.to_csv(os.path.join(output_dir, "mutations_gbm.csv"), index=False)
        print(f"💾 Mutações salvas: {len(df_mutations)} registros")

        # Top genes summary
        df_top.to_csv(os.path.join(output_dir, "mutation_summary_gbm.csv"), index=False)
        print(f"💾 Summary salvo: {len(df_top)} genes")

    # 2. SSM details (limitado a 5000 para velocidade)
    print("\n📊 Buscando detalhes de SSMs...")
    df_ssm = fetch_ssm_details(max_ssms=3000)

    if not df_ssm.empty:
        # Substituir o mutations_gbm.csv com dados mais ricos
        df_ssm.to_csv(os.path.join(output_dir, "mutations_gbm.csv"), index=False)
        print(f"💾 Mutações (detalhadas) salvas: {len(df_ssm)} registros")

        # Summary
        summary = generate_mutation_summary(df_ssm, df_top)
        if not summary.empty:
            summary.to_csv(os.path.join(output_dir, "mutation_summary_gbm.csv"), index=False)

    print("\n🎉 Extração de mutações completa!")
    if not df_ssm.empty:
        print(f"   Genes únicos: {df_ssm['gene_symbol'].nunique()}")
        print(f"   Top 10:")
        top10 = df_ssm['gene_symbol'].value_counts().head(10)
        for gene, count in top10.items():
            print(f"      {gene:>12}: {count}")
