"""
prescriptive.py — Motor de Análise Prescritiva para GBM

Funcionalidades:
  1. Patient Risk Calculator — score de risco individual
  2. Treatment Recommendation — recomendação baseada em pacientes similares
  3. Gene Alert System — flags automáticas de combinações de risco
"""
import os, json
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reports")

def load_resources():
    """Carrega modelo e dados clínicos."""
    model_path = "models/modelo_sobrevivencia_gbm_v2.pkl"
    if not os.path.exists(model_path):
        model_path = "models/modelo_sobrevivencia_gbm_v1.pkl"
    pkg = joblib.load(model_path)
    df_clinical = pd.read_csv("data/processed/clinical_gbm_completo.csv")
    return pkg, df_clinical

def calculate_risk_score(model, patient_features):
    """Calcula probabilidade de alto risco para um paciente."""
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(patient_features.values.reshape(1, -1))[0]
        return {'risk_score': float(proba[1]), 'class': 'Alto Risco' if proba[1] > 0.5 else 'Baixo Risco',
                'confidence': float(max(proba))}
    pred = model.predict(patient_features.values.reshape(1, -1))[0]
    return {'risk_score': float(pred), 'class': 'Alto Risco' if pred == 1 else 'Baixo Risco', 'confidence': None}

def find_similar_patients(df_clinical, patient_age, patient_gender, n_neighbors=10):
    """Encontra pacientes historicamente similares baseado em perfil clínico."""
    df = df_clinical.copy()
    df['age_diff'] = abs(df['idade'] - patient_age)
    df_same_gender = df[df['genero'] == patient_gender] if patient_gender else df
    if len(df_same_gender) < n_neighbors:
        df_same_gender = df
    similar = df_same_gender.nsmallest(n_neighbors, 'age_diff')
    return similar

def treatment_recommendation(df_clinical, patient_age, patient_gender, risk_class):
    """Recomenda tratamento baseado em outcomes de pacientes similares."""
    similar = find_similar_patients(df_clinical, patient_age, patient_gender, n_neighbors=30)
    # Filtrar por mesma classe de risco com bom outcome
    if risk_class == 'Alto Risco':
        # Buscar alto risco que sobreviveram mais
        survivors = similar[similar['dias_sobrevivencia'] > 365]
    else:
        survivors = similar[similar['dias_sobrevivencia'] > 730]

    if survivors.empty:
        survivors = similar.nlargest(10, 'dias_sobrevivencia')

    # Contar tratamentos dos sobreviventes
    treatments = {}
    for _, row in survivors.iterrows():
        for t in str(row.get('tipos_tratamento', '')).split(' | '):
            t = t.strip()
            if t and t != 'Não Informado':
                treatments[t] = treatments.get(t, 0) + 1

    # Ordenar por frequência
    sorted_treatments = sorted(treatments.items(), key=lambda x: x[1], reverse=True)
    recs = []
    for treat, count in sorted_treatments[:5]:
        pct = count / len(survivors) * 100
        recs.append({'treatment': treat, 'frequency': count, 'pct_survivors': round(pct, 1)})
    return recs

def gene_alert_system(shap_path="data/reports/shap_feature_importance.csv", threshold=0.01):
    """Identifica genes com alto impacto no risco."""
    if not os.path.exists(shap_path):
        return {'risk_genes': [], 'protective_genes': []}
    df = pd.read_csv(shap_path)
    risk = df[(df['direction'] == 'RISCO') & (df['mean_abs_shap'] > threshold)].head(20)
    protective = df[(df['direction'] == 'PROTETOR') & (df['mean_abs_shap'] > threshold)].head(20)
    return {
        'risk_genes': risk[['feature', 'mean_shap', 'mean_abs_shap']].to_dict('records'),
        'protective_genes': protective[['feature', 'mean_shap', 'mean_abs_shap']].to_dict('records')
    }

def generate_prescriptive_report():
    """Gera relatório prescritivo completo."""
    print("🏥 Gerando Relatório Prescritivo...")
    pkg, df_clinical = load_resources()
    alerts = gene_alert_system()

    # Estatísticas de tratamento
    treatment_stats = {}
    for _, row in df_clinical.iterrows():
        for t in str(row.get('tipos_tratamento', '')).split(' | '):
            t = t.strip()
            if t and t != 'Não Informado':
                if t not in treatment_stats:
                    treatment_stats[t] = {'count': 0, 'survival_days': [], 'high_risk': 0, 'low_risk': 0}
                treatment_stats[t]['count'] += 1
                treatment_stats[t]['survival_days'].append(row['dias_sobrevivencia'])
                if row['classe_risco'] == 'Alto Risco':
                    treatment_stats[t]['high_risk'] += 1
                else:
                    treatment_stats[t]['low_risk'] += 1

    # Calcular medianas
    for t in treatment_stats:
        days = treatment_stats[t]['survival_days']
        treatment_stats[t]['median_survival'] = float(np.median(days))
        treatment_stats[t]['mean_survival'] = float(np.mean(days))
        del treatment_stats[t]['survival_days']

    report = {
        'total_patients': len(df_clinical),
        'model_name': pkg.get('model_name', 'Unknown'),
        'gene_alerts': alerts,
        'treatment_analysis': treatment_stats,
        'risk_distribution': {
            'alto_risco': int((df_clinical['classe_risco'] == 'Alto Risco').sum()),
            'baixo_risco': int((df_clinical['classe_risco'] == 'Baixo Risco').sum()),
        }
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, 'prescriptive_report.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"💾 Relatório salvo: {path}")
    return report

if __name__ == "__main__":
    report = generate_prescriptive_report()
    print(f"\n📊 Pacientes: {report['total_patients']}")
    print(f"🔴 Alto Risco: {report['risk_distribution']['alto_risco']}")
    print(f"🟢 Baixo Risco: {report['risk_distribution']['baixo_risco']}")
    print(f"🧬 Genes de Risco: {len(report['gene_alerts']['risk_genes'])}")
    print(f"🛡️  Genes Protetores: {len(report['gene_alerts']['protective_genes'])}")
