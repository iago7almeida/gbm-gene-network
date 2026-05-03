"""
explainability.py — Interpretabilidade de Modelos com SHAP

Gera: SHAP summary plots, dependence plots, force plots,
rankings de genes de risco vs protetores.
"""
import os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

def load_model_package(model_path="models/modelo_sobrevivencia_gbm_v2.pkl"):
    """Carrega o pacote completo do modelo (model + preprocessadores)."""
    pkg = joblib.load(model_path)
    print(f"📂 Modelo carregado: {pkg.get('model_name', 'Unknown')}")
    return pkg

def compute_shap_values(model, X, model_name="XGBoost"):
    """Calcula SHAP values para o modelo tree-based."""
    print(f"\n🔍 Calculando SHAP values para {model_name}...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # Para classificação binária, pegar valores da classe positiva
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        print(f"✅ SHAP values calculados: {shap_values.shape}")
        return explainer, shap_values
    except Exception as e:
        print(f"⚠️  TreeExplainer falhou ({e}), tentando KernelExplainer...")
        bg = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(X, nsamples=200)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return explainer, shap_values

def generate_shap_plots(shap_values, X, feature_names=None, top_n=30):
    """Gera todos os plots SHAP e salva como PNG."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if feature_names is not None:
        X_display = X.copy()
        X_display.columns = feature_names[:len(X_display.columns)]

    # 1. Summary Plot (Beeswarm)
    print("📊 Gerando SHAP Summary Plot...")
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))
    shap.summary_plot(shap_values, X, max_display=top_n, show=False, plot_size=None)
    plt.title('SHAP Feature Importance — Top Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close('all')

    # 2. Bar Plot (Mean |SHAP|)
    print("📊 Gerando SHAP Bar Plot...")
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.25)))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=top_n, show=False, plot_size=None)
    plt.title('Mean |SHAP| — Feature Importance Global', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close('all')

    print(f"✅ Plots salvos em {FIGURES_DIR}")

def classify_risk_genes(shap_values, feature_names, top_n=50):
    """
    Classifica genes como de RISCO ou PROTETORES baseado no SHAP.
    SHAP positivo → contribui para Alto Risco
    SHAP negativo → contribui para Baixo Risco (protetor)
    """
    mean_shap = np.mean(shap_values, axis=0)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    df = pd.DataFrame({
        'feature': feature_names[:len(mean_shap)],
        'mean_shap': mean_shap,
        'mean_abs_shap': mean_abs_shap,
        'direction': ['RISCO' if s > 0 else 'PROTETOR' for s in mean_shap]
    }).sort_values('mean_abs_shap', ascending=False)

    risk_genes = df[df['direction'] == 'RISCO'].head(top_n // 2)
    protective_genes = df[df['direction'] == 'PROTETOR'].head(top_n // 2)

    print(f"\n🔴 Top {len(risk_genes)} Genes/Features de RISCO (SHAP > 0):")
    for _, r in risk_genes.head(10).iterrows():
        print(f"   {r['feature']:>20}: SHAP = {r['mean_shap']:+.4f}")

    print(f"\n🟢 Top {len(protective_genes)} Genes/Features PROTETORES (SHAP < 0):")
    for _, r in protective_genes.head(10).iterrows():
        print(f"   {r['feature']:>20}: SHAP = {r['mean_shap']:+.4f}")

    return df

def run_explainability_pipeline(model_path="models/modelo_sobrevivencia_gbm_v2.pkl",
                                 clinical_path="data/processed/clinical_gbm_completo.csv",
                                 expression_path="data/raw/matriz_gbm_bruta.csv",
                                 nodes_path="data/processed/gbm_genes_nodes.csv"):
    """Pipeline completo de interpretabilidade."""
    # Tentar carregar v2, fallback para v1
    if not os.path.exists(model_path):
        model_path = "models/modelo_sobrevivencia_gbm_v1.pkl"
    if not os.path.exists(model_path):
        print("❌ Nenhum modelo encontrado. Execute train.py primeiro.")
        return

    pkg = load_model_package(model_path)
    model = pkg['model']
    feature_names = pkg.get('feature_names', [])

    # Reconstruir X usando o mesmo pipeline do treino
    from train import load_and_merge_data, preprocess_data
    df = load_and_merge_data(clinical_path, expression_path, nodes_path)
    if df.empty: return

    X, y, _, _, _ = preprocess_data(df)

    # Calcular SHAP
    _, shap_vals = compute_shap_values(model, X, pkg.get('model_name', 'Model'))

    # Gerar plots
    generate_shap_plots(shap_vals, X, feature_names)

    # Classificar genes
    gene_risk_df = classify_risk_genes(shap_vals, list(X.columns))

    # Salvar resultados
    os.makedirs(REPORTS_DIR, exist_ok=True)
    shap_path = os.path.join(REPORTS_DIR, 'shap_feature_importance.csv')
    gene_risk_df.to_csv(shap_path, index=False)
    print(f"\n💾 SHAP values salvos em: {shap_path}")

    # Salvar SHAP values completos para o dashboard
    shap_full = pd.DataFrame(shap_vals, columns=X.columns)
    shap_full.to_csv(os.path.join(REPORTS_DIR, 'shap_values_full.csv'), index=False)

    return gene_risk_df

if __name__ == "__main__":
    run_explainability_pipeline()
