"""
survival_analysis.py — Análise de Sobrevivência Profissional (GBM)

Implementa: Kaplan-Meier, Cox PH, Log-rank test, C-Index
Biblioteca: lifelines
"""
import os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

def load_clinical_data(path="data/processed/clinical_gbm_completo.csv"):
    df = pd.read_csv(path)
    df['event_observed'] = ~df['is_censored']
    df['dias_sobrevivencia'] = pd.to_numeric(df['dias_sobrevivencia'], errors='coerce')
    df = df.dropna(subset=['dias_sobrevivencia'])
    df = df[df['dias_sobrevivencia'] > 0]
    df['faixa_etaria'] = pd.cut(df['idade'], bins=[0,40,55,65,100], labels=['<40','40-55','55-65','>65'])
    print(f"📊 Dados: {len(df)} pacientes | Eventos: {df['event_observed'].sum()} | Censurados: {(~df['event_observed']).sum()}")
    return df

def kaplan_meier_overall(df):
    print("\n📈 Kaplan-Meier geral...")
    kmf = KaplanMeierFitter()
    kmf.fit(df['dias_sobrevivencia'], df['event_observed'], label='GBM Overall')
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel('Dias'); ax.set_ylabel('Prob. Sobrevivência')
    ax.set_title('Kaplan-Meier — Coorte GBM', fontsize=14, fontweight='bold')
    median_s = kmf.median_survival_time_
    if not np.isinf(median_s):
        ax.axvline(x=median_s, color='red', linestyle=':', alpha=0.7)
        ax.text(median_s+20, 0.52, f'Mediana: {median_s:.0f}d', fontsize=10, color='red')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, 'km_overall.png'), dpi=150); plt.close()
    print(f"   Mediana: {median_s:.0f} dias")
    return kmf

def kaplan_meier_by_group(df, group_col, max_groups=6):
    print(f"\n📈 KM por {group_col}...")
    df_c = df.dropna(subset=[group_col]).copy()
    groups = df_c[group_col].value_counts().head(max_groups).index.tolist()
    df_c = df_c[df_c[group_col].isin(groups)]
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    km_results = {}
    for i, g in enumerate(groups):
        m = df_c[group_col] == g
        kmf = KaplanMeierFitter()
        kmf.fit(df_c.loc[m, 'dias_sobrevivencia'], df_c.loc[m, 'event_observed'], label=f'{g} (n={m.sum()})')
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[i])
        med = float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else None
        km_results[str(g)] = {'median': med, 'n': int(m.sum()), 'events': int(df_c.loc[m,'event_observed'].sum())}
    ax.set_xlabel('Dias'); ax.set_ylabel('Prob. Sobrevivência')
    ax.set_title(f'Kaplan-Meier por {group_col}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f'km_by_{group_col.lower().replace(" ","_")}.png'), dpi=150); plt.close()
    if len(groups) == 2:
        g1, g2 = df_c[group_col]==groups[0], df_c[group_col]==groups[1]
        lr = logrank_test(df_c.loc[g1,'dias_sobrevivencia'], df_c.loc[g2,'dias_sobrevivencia'],
                          event_observed_A=df_c.loc[g1,'event_observed'], event_observed_B=df_c.loc[g2,'event_observed'])
        print(f"   Log-rank p={lr.p_value:.6f} {'✅ Sig.' if lr.p_value<0.05 else '⚠️ NS'}")
        km_results['logrank_p'] = float(lr.p_value)
    elif len(groups) > 2:
        try:
            lr = multivariate_logrank_test(df_c['dias_sobrevivencia'], df_c[group_col], df_c['event_observed'])
            km_results['logrank_p'] = float(lr.p_value)
        except: pass
    return km_results

def cox_proportional_hazards(df):
    print("\n📊 Cox Proportional Hazards...")
    cox_df = df[['dias_sobrevivencia','event_observed','idade']].copy()
    if 'genero' in df.columns: cox_df['is_male'] = (df['genero']=='male').astype(int)
    if 'classe_risco' in df.columns: cox_df['alto_risco'] = (df['classe_risco']=='Alto Risco').astype(int)
    cox_df = cox_df.dropna()
    cph = CoxPHFitter(penalizer=0.01)
    try:
        cph.fit(cox_df, duration_col='dias_sobrevivencia', event_col='event_observed', show_progress=False)
        cph.print_summary()
        c_idx = cph.concordance_index_
        print(f"\n🏆 C-Index: {c_idx:.4f}")
        hr_df = pd.DataFrame({
            'covariate': cph.summary.index, 'hazard_ratio': cph.hazard_ratios_.values,
            'p_value': cph.summary['p'].values,
            'ci_lower': np.exp(cph.summary['coef lower 95%'].values),
            'ci_upper': np.exp(cph.summary['coef upper 95%'].values),
        })
        # Forest plot
        fig, ax = plt.subplots(figsize=(10, max(4, len(hr_df)*0.6+1)))
        y = range(len(hr_df))
        cols = ['#e74c3c' if h>1 else '#27ae60' for h in hr_df['hazard_ratio']]
        ax.barh(y, hr_df['hazard_ratio']-1, left=1, color=cols, alpha=0.7, height=0.6)
        ax.errorbar(hr_df['hazard_ratio'], y,
                     xerr=[hr_df['hazard_ratio']-hr_df['ci_lower'], hr_df['ci_upper']-hr_df['hazard_ratio']],
                     fmt='o', color='black', markersize=6, capsize=3)
        ax.axvline(x=1, color='gray', linestyle='--')
        ax.set_yticks(y); ax.set_yticklabels(hr_df['covariate'], fontsize=11)
        ax.set_xlabel('Hazard Ratio'); ax.set_title('Forest Plot — Cox PH', fontsize=14, fontweight='bold')
        for i, r in hr_df.iterrows():
            sig = '***' if r['p_value']<0.001 else '**' if r['p_value']<0.01 else '*' if r['p_value']<0.05 else 'ns'
            ax.text(max(hr_df['ci_upper'])*1.05, i, f"HR={r['hazard_ratio']:.2f} ({sig})", fontsize=9, va='center')
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, 'cox_forest_plot.png'), dpi=150); plt.close()
        return cph, hr_df, c_idx
    except Exception as e:
        print(f"❌ Cox PH erro: {e}")
        return None, pd.DataFrame(), 0.0

def survival_analysis_report(df):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    report = {}
    kmf = kaplan_meier_overall(df)
    report['median_survival'] = float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else None
    report['km_risk'] = kaplan_meier_by_group(df, 'classe_risco')
    report['km_age'] = kaplan_meier_by_group(df, 'faixa_etaria')
    report['km_gender'] = kaplan_meier_by_group(df, 'genero')
    cph, hr_df, c_idx = cox_proportional_hazards(df)
    report['cox_c_index'] = c_idx
    if not hr_df.empty: report['hazard_ratios'] = hr_df.to_dict(orient='records')
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, 'survival_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    if not hr_df.empty: hr_df.to_csv(os.path.join(REPORTS_DIR, 'hazard_ratios.csv'), index=False)
    print(f"\n💾 Relatório salvo em {REPORTS_DIR}")
    return report

if __name__ == "__main__":
    df = load_clinical_data()
    report = survival_analysis_report(df)
    print(f"\n🎉 Mediana: {report.get('median_survival','N/A')}d | C-Index: {report.get('cox_c_index',0):.4f}")
