import requests
import json
import pandas as pd
import os

def fetch_clinical_data(project_id="TCGA-GBM"):
    """Consulta a API do GDC para baixar os dados clínicos completos dos pacientes."""
    cases_endpoint = "https://api.gdc.cancer.gov/cases"
    
    filters = {
        "op": "in",
        "content": {"field": "project.project_id", "value": [project_id]}
    }
    
    # Lista de campos expandida com as correções de sintaxe
    fields = [
        "submitter_id",
        "demographic.vital_status",
        "demographic.days_to_death",
        "demographic.days_to_last_follow_up",
        "demographic.age_at_index",
        "demographic.gender",
        "demographic.race",
        "demographic.ethnicity",
        "diagnoses.primary_diagnosis",
        "diagnoses.tumor_stage",
        "diagnoses.grade",
        "diagnoses.morphology",
        "diagnoses.site_of_resection_or_biopsy",
        "diagnoses.days_to_diagnosis",
        "diagnoses.treatments.treatment_type",
        "diagnoses.treatments.days_to_treatment_start",
        "diagnoses.treatments.therapeutic_agents",
        "diagnoses.tumor_grade",
        "diagnoses.classification_of_tumor"
    ]
    
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "1000" 
    }
    
    print("🔍 Consultando a API do GDC pelo escopo clínico completo...")
    response = requests.get(cases_endpoint, params=params)
    response.raise_for_status()
    
    return response.json()["data"]["hits"]

def clean_clinical_data(hits):
    """Achata o JSON e cria uma matriz rica (Features + Target) para Machine Learning."""
    print("🧹 Estruturando variáveis demográficas, diagnósticas e alvo...")
    records = []
    
    for hit in hits:
        submitter_id = hit.get("submitter_id")
        demo = hit.get("demographic", {})
        
        # O GDC retorna diagnósticos como uma lista. Vamos focar no diagnóstico primário [0]
        diagnoses_list = hit.get("diagnoses", [])
        diag = diagnoses_list[0] if diagnoses_list else {}
        
        # 1. Variável Alvo (Sobrevivência)
        status = demo.get("vital_status")
        if status is None:
            continue
            
        survival_days = demo.get("days_to_death") if status == "Dead" else demo.get("days_to_last_follow_up")
        
        # 2. Tratamentos (Pode haver múltiplos, então vamos extrair os tipos numa string única)
        treatments_list = diag.get("treatments", [])
        treatment_types = [t.get("treatment_type") for t in treatments_list if t.get("treatment_type")]
        treatments_str = " | ".join(treatment_types) if treatment_types else "Não Informado"
        
        # Montando a linha do paciente com o escopo ampliado
        records.append({
            "paciente_id": submitter_id,
            "status_vital": status,
            "dias_sobrevivencia": survival_days,
            
            # Demografia (Features)
            "idade": demo.get("age_at_index"),
            "genero": demo.get("gender"),
            "raca": demo.get("race"),
            "etnia": demo.get("ethnicity"),
            
            # Diagnóstico (Features)
            "diagnostico_primario": diag.get("primary_diagnosis"),
            "estagio_tumor": diag.get("tumor_stage"),
            "grau_tumor": diag.get("tumor_grade", diag.get("grade")), # Fallback entre os dois campos de grau
            "morfologia": diag.get("morphology"),
            "local_biopsia": diag.get("site_of_resection_or_biopsy"),
            "dias_para_diagnostico": diag.get("days_to_diagnosis"),
            
            # Tratamento (Features)
            "tipos_tratamento": treatments_str
        })
            
    df_clinical = pd.DataFrame(records)

    # Removendo colunas que não se aplicam ao contexto do Glioblastoma ou estão vazias
    colunas_para_remover = ['estagio_tumor', 'grau_tumor']
    df_clinical = df_clinical.drop(columns=colunas_para_remover, errors='ignore')
    
    # Limpeza final e criação da classe de risco
    df_clinical = df_clinical.dropna(subset=["dias_sobrevivencia", "idade"])
    df_clinical['classe_risco'] = df_clinical['dias_sobrevivencia'].apply(
        lambda x: 'Alto Risco' if x < 365 else 'Baixo Risco'
    )
    
    print(f"✅ Dados processados com sucesso! Matriz gerada com {df_clinical.shape[1]} colunas para {df_clinical.shape[0]} pacientes.")
    return df_clinical

if __name__ == "__main__":
    hits = fetch_clinical_data()
    df_clin = clean_clinical_data(hits)
    
    output_path = "data/processed/clinical_gbm_completo.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clin.to_csv(output_path, index=False)
    
    print(f"💾 Base clínica salva em: {output_path}")