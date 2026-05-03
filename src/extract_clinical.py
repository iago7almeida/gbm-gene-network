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
    """
    Achata o JSON e cria uma matriz rica (Features + Target) para Machine Learning.

    CORREÇÃO CRÍTICA: Pacientes vivos (Alive) são agora incluídos com a flag
    'is_censored=True'. Excluí-los anteriormente introduzia viés de sobrevivência
    severo — a classe 'Baixo Risco' ficava subrepresentada pois os sobreviventes
    de longa data eram removidos do dataset.
    """
    print("🧹 Estruturando variáveis demográficas, diagnósticas e alvo...")
    records = []
    
    skipped_no_status = 0
    skipped_no_survival = 0
    included_dead = 0
    included_alive = 0

    for hit in hits:
        submitter_id = hit.get("submitter_id")
        demo = hit.get("demographic", {})
        
        # O GDC retorna diagnósticos como uma lista. Vamos focar no diagnóstico primário [0]
        diagnoses_list = hit.get("diagnoses", [])
        diag = diagnoses_list[0] if diagnoses_list else {}
        
        # 1. Variável Alvo (Sobrevivência)
        status = demo.get("vital_status")
        if status is None:
            skipped_no_status += 1
            continue

        # CORREÇÃO: Pacientes vivos usam days_to_last_follow_up como tempo de censura.
        # Pacientes mortos usam days_to_death como tempo de evento.
        if status == "Dead":
            survival_days = demo.get("days_to_death")
            is_censored = False
        else:  # "Alive" — tempo de seguimento, observação censurada
            survival_days = demo.get("days_to_last_follow_up")
            is_censored = True

        # Descarta apenas quem não tem nenhuma informação de tempo de sobrevivência
        if survival_days is None:
            skipped_no_survival += 1
            continue

        # 2. Tratamentos — ordenar alfabeticamente antes de unir para evitar duplicatas no OHE
        # Ex: "A | B" e "B | A" representam o mesmo tratamento mas geravam features diferentes
        treatments_list = diag.get("treatments", [])
        treatment_types = sorted([
            t.get("treatment_type") for t in treatments_list
            if t.get("treatment_type")
        ])
        treatments_str = " | ".join(treatment_types) if treatment_types else "Não Informado"

        records.append({
            "paciente_id": submitter_id,
            "status_vital": status,
            "is_censored": is_censored,          # Nova flag: True = paciente ainda vivo
            "dias_sobrevivencia": survival_days,

            # Demografia (Features)
            "idade": demo.get("age_at_index"),
            "genero": demo.get("gender"),
            "raca": demo.get("race"),
            "etnia": demo.get("ethnicity"),

            # Diagnóstico (Features)
            "diagnostico_primario": diag.get("primary_diagnosis"),
            "morfologia": diag.get("morphology"),
            "local_biopsia": diag.get("site_of_resection_or_biopsy"),
            "dias_para_diagnostico": diag.get("days_to_diagnosis"),

            # Tratamento (Features)
            "tipos_tratamento": treatments_str
        })

        if status == "Dead":
            included_dead += 1
        else:
            included_alive += 1

    df_clinical = pd.DataFrame(records)

    # Limpeza final — remove apenas quem não tem idade (essencial para o modelo)
    df_clinical = df_clinical.dropna(subset=["dias_sobrevivencia", "idade"])

    # Criação da classe de risco binária (corte padrão em 1 ano = 365 dias)
    # Pacientes vivos com follow-up > 365 dias são conservadoramente classificados como Baixo Risco
    df_clinical['classe_risco'] = df_clinical['dias_sobrevivencia'].apply(
        lambda x: 'Alto Risco' if x < 365 else 'Baixo Risco'
    )

    print(f"\n📊 Resumo da extração clínica:")
    print(f"   ✅ Pacientes MORTOS incluídos  : {included_dead}")
    print(f"   ✅ Pacientes VIVOS incluídos   : {included_alive}")
    print(f"   ⚠️  Sem status vital (ignorados): {skipped_no_status}")
    print(f"   ⚠️  Sem tempo de sobrevivência : {skipped_no_survival}")
    print(f"   📐 Shape final                 : {df_clinical.shape}")
    print(f"   🎯 Alto Risco  (<365 dias)     : {(df_clinical['classe_risco'] == 'Alto Risco').sum()}")
    print(f"   🎯 Baixo Risco (≥365 dias)     : {(df_clinical['classe_risco'] == 'Baixo Risco').sum()}")
    return df_clinical

if __name__ == "__main__":
    hits = fetch_clinical_data()
    df_clin = clean_clinical_data(hits)

    output_path = "data/processed/clinical_gbm_completo.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clin.to_csv(output_path, index=False)

    print(f"\n💾 Base clínica salva em: {output_path}")