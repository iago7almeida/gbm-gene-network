"""
GBM Gene Network — Dashboard Interativo Profissional
Multi-page Dash app com storytelling de dados oncológicos.
"""
import os, sys, json
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# ============================================================
# DATA LOADING
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def safe_load_csv(path, default_cols=None):
    fp = os.path.join(BASE, path)
    if os.path.exists(fp):
        return pd.read_csv(fp)
    return pd.DataFrame(columns=default_cols or [])

def safe_load_json(path):
    fp = os.path.join(BASE, path)
    if os.path.exists(fp):
        with open(fp) as f:
            return json.load(f)
    return {}

df_clinical = safe_load_csv("data/processed/clinical_gbm_completo.csv")
df_shap = safe_load_csv("data/reports/shap_feature_importance.csv", ["feature","mean_shap","mean_abs_shap","direction"])
df_mutations = safe_load_csv("data/processed/mutations_gbm.csv", ["paciente_id","gene_symbol","consequence_type"])
df_hazard = safe_load_csv("data/reports/hazard_ratios.csv", ["covariate","hazard_ratio","p_value"])
model_meta = safe_load_json("models/modelo_metadata_v2.json") or safe_load_json("models/modelo_metadata_v1.json")
survival_report = safe_load_json("data/reports/survival_report.json")
prescriptive = safe_load_json("data/reports/prescriptive_report.json")
comparison = safe_load_json("models/model_comparison_report.json")

PLOT_TEMPLATE = "plotly_dark"
COLORS = {"risk": "#f43f5e", "safe": "#10b981", "blue": "#3b82f6", "purple": "#8b5cf6",
          "cyan": "#06b6d4", "amber": "#f59e0b", "bg": "#111827"}

# ============================================================
# APP INIT
# ============================================================
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="GBM Oncology Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# ============================================================
# SIDEBAR
# ============================================================
def make_nav(icon, label, page_id):
    return html.Div([html.Span(icon, className="nav-icon"), html.Span(label)],
                    id=f"nav-{page_id}", className="nav-link", n_clicks=0)

sidebar = html.Div([
    html.Div([
        html.Div("🧠", className="brand-icon"),
        html.Div([html.H2("GBM Network"), html.Small("Oncology Analytics")]),
    ], className="sidebar-brand"),
    html.Div([
        html.Div("Analytics", className="nav-section-title"),
        make_nav("📊", "Overview", "overview"),
        make_nav("📈", "Sobrevivência", "survival"),
        make_nav("🧬", "Risco Genômico", "genomic"),
        make_nav("🔬", "Mutações", "mutations"),
    ], className="nav-section"),
    html.Div([
        html.Div("Insights", className="nav-section-title"),
        make_nav("💊", "Tratamentos", "treatment"),
        make_nav("🕸️", "Rede Gênica", "network"),
        make_nav("🏥", "Prescrição", "prescriptive"),
    ], className="nav-section"),
], className="sidebar")

# ============================================================
# KPI HELPERS
# ============================================================
def kpi_card(label, value, subtitle="", color="blue"):
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(str(value), className="kpi-value"),
        html.Div(subtitle, className="kpi-subtitle"),
    ], className=f"kpi-card {color}")

# ============================================================
# PAGE: OVERVIEW
# ============================================================
def page_overview():
    n_patients = len(df_clinical)
    n_high = int((df_clinical['classe_risco']=='Alto Risco').sum()) if not df_clinical.empty else 0
    n_low = n_patients - n_high
    median_surv = df_clinical['dias_sobrevivencia'].median() if not df_clinical.empty else 0
    best_model = model_meta.get('model_name', 'N/A')
    best_auc = "N/A"
    if comparison.get('cv_results'):
        best_auc_val = max(v.get('roc_auc',0) for v in comparison['cv_results'].values())
        best_auc = f"{best_auc_val:.3f}"

    kpis = html.Div([
        kpi_card("Total Pacientes", n_patients, "TCGA-GBM Cohort", "blue"),
        kpi_card("Alto Risco", n_high, f"{n_high/max(n_patients,1)*100:.0f}% da coorte", "red"),
        kpi_card("Mediana Sobreviv.", f"{median_surv:.0f}d", f"≈ {median_surv/30:.0f} meses", "purple"),
        kpi_card("Melhor ROC-AUC", best_auc, best_model, "green"),
    ], className="kpi-grid")

    # Risk distribution
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Pie(labels=["Alto Risco","Baixo Risco"], values=[n_high, n_low],
                              hole=0.65, marker_colors=[COLORS["risk"], COLORS["safe"]],
                              textinfo="label+percent", textfont_size=14))
    fig_risk.update_layout(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=350,
                           margin=dict(t=30,b=30,l=30,r=30))

    # Age distribution
    fig_age = go.Figure()
    if not df_clinical.empty:
        for cls, color in [("Alto Risco", COLORS["risk"]), ("Baixo Risco", COLORS["safe"])]:
            subset = df_clinical[df_clinical['classe_risco']==cls]
            fig_age.add_trace(go.Histogram(x=subset['idade'], name=cls, marker_color=color, opacity=0.7, nbinsx=20))
    fig_age.update_layout(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", barmode="overlay", height=350,
                          xaxis_title="Idade", yaxis_title="Pacientes", margin=dict(t=30,b=50,l=50,r=30))

    # Model comparison
    fig_models = go.Figure()
    if comparison.get('cv_results'):
        names = list(comparison['cv_results'].keys())
        aucs = [comparison['cv_results'][n].get('roc_auc',0) for n in names]
        fig_models.add_trace(go.Bar(x=aucs, y=names, orientation='h',
                                    marker_color=[COLORS["blue"] if a==max(aucs) else COLORS["purple"] for a in aucs],
                                    text=[f"{a:.3f}" for a in aucs], textposition='outside'))
    fig_models.update_layout(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", height=350, xaxis_title="ROC-AUC (CV)",
                             margin=dict(t=30,b=50,l=150,r=60))

    # Survival by gender
    fig_gender = go.Figure()
    if not df_clinical.empty and 'genero' in df_clinical.columns:
        for g, c in [("male", COLORS["blue"]), ("female", COLORS["purple"])]:
            s = df_clinical[df_clinical['genero']==g]
            fig_gender.add_trace(go.Box(y=s['dias_sobrevivencia'], name=g.capitalize(), marker_color=c))
    fig_gender.update_layout(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", height=350, yaxis_title="Dias Sobrevivência",
                             margin=dict(t=30,b=50,l=60,r=30))

    return html.Div([
        html.Div([html.H1("Executive Overview"), html.P("Visão geral da coorte TCGA-GBM e performance dos modelos.")], className="page-header"),
        kpis,
        html.Div([
            html.Div([html.Div("Distribuição de Risco", className="card-title"),
                       dcc.Graph(figure=fig_risk, config={"displayModeBar": False})], className="card"),
            html.Div([html.Div("Distribuição Etária por Risco", className="card-title"),
                       dcc.Graph(figure=fig_age, config={"displayModeBar": False})], className="card"),
            html.Div([html.Div("Comparação de Modelos (CV)", className="card-title"),
                       dcc.Graph(figure=fig_models)], className="card"),
            html.Div([html.Div("Sobrevivência por Gênero", className="card-title"),
                       dcc.Graph(figure=fig_gender)], className="card"),
        ], className="chart-grid"),
    ], className="fade-in")

# ============================================================
# PAGE: SURVIVAL
# ============================================================
def page_survival():
    # KM-like visualization using clinical data
    fig_km = go.Figure()
    if not df_clinical.empty:
        for cls, color in [("Alto Risco", COLORS["risk"]), ("Baixo Risco", COLORS["safe"])]:
            subset = df_clinical[df_clinical['classe_risco']==cls].sort_values('dias_sobrevivencia')
            n = len(subset)
            times = subset['dias_sobrevivencia'].values
            surv = np.arange(n, 0, -1) / n
            fig_km.add_trace(go.Scatter(x=times, y=surv, mode='lines', name=f"{cls} (n={n})",
                                        line=dict(color=color, width=2.5), fill='tozeroy',
                                        fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.1)"))
    fig_km.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig_km.update_layout(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         height=500, xaxis_title="Dias", yaxis_title="Probabilidade de Sobrevivência",
                         margin=dict(t=30,b=60,l=60,r=30))

    # Hazard ratios
    fig_hr = go.Figure()
    if not df_hazard.empty:
        fig_hr.add_trace(go.Bar(x=df_hazard['hazard_ratio'], y=df_hazard['covariate'], orientation='h',
                                marker_color=[COLORS["risk"] if h>1 else COLORS["safe"] for h in df_hazard['hazard_ratio']],
                                text=[f"{h:.2f}" for h in df_hazard['hazard_ratio']], textposition='outside'))
        fig_hr.add_vline(x=1, line_dash="dash", line_color="white", opacity=0.3)
    fig_hr.update_layout(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         height=350, xaxis_title="Hazard Ratio", margin=dict(t=30,b=50,l=120,r=60))

    c_idx = survival_report.get('cox_c_index', 'N/A')
    median_s = survival_report.get('median_survival', 'N/A')

    return html.Div([
        html.Div([html.H1("Análise de Sobrevivência"), html.P("Curvas de Kaplan-Meier, Cox PH e hazard ratios.")], className="page-header"),
        html.Div([
            kpi_card("Mediana Geral", f"{median_s}d" if median_s != 'N/A' else "N/A", "Kaplan-Meier", "purple"),
            kpi_card("C-Index", f"{c_idx:.3f}" if isinstance(c_idx, float) else c_idx, "Cox PH", "blue"),
            kpi_card("Eventos", f"{df_clinical['is_censored'].eq(False).sum() if not df_clinical.empty else 0}", "Mortes observadas", "red"),
            kpi_card("Censurados", f"{df_clinical['is_censored'].eq(True).sum() if not df_clinical.empty else 0}", "Ainda vivos", "green"),
        ], className="kpi-grid"),
        html.Div([
            html.Div([html.Div("Curva de Sobrevivência por Classe de Risco", className="card-title"),
                       dcc.Graph(figure=fig_km)], className="card full-width"),
            html.Div([html.Div("Forest Plot — Hazard Ratios (Cox PH)", className="card-title"),
                       dcc.Graph(figure=fig_hr)], className="card full-width"),
        ], className="chart-grid"),
    ], className="fade-in")

# ============================================================
# PAGE: GENOMIC RISK
# ============================================================
def page_genomic():
    fig_shap = go.Figure()
    if not df_shap.empty:
        top = df_shap.nlargest(25, 'mean_abs_shap')
        colors = [COLORS["risk"] if d=="RISCO" else COLORS["safe"] for d in top['direction']]
        fig_shap.add_trace(go.Bar(x=top['mean_shap'], y=top['feature'], orientation='h',
                                   marker_color=colors, text=[f"{s:+.4f}" for s in top['mean_shap']], textposition='outside'))
    fig_shap.update_layout(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           height=600, xaxis_title="Mean SHAP Value", yaxis=dict(autorange="reversed"),
                           margin=dict(t=30,b=50,l=160,r=80))

    n_risk = len(df_shap[df_shap['direction']=='RISCO']) if not df_shap.empty else 0
    n_prot = len(df_shap[df_shap['direction']=='PROTETOR']) if not df_shap.empty else 0

    table_data = df_shap.head(30).to_dict('records') if not df_shap.empty else []

    return html.Div([
        html.Div([html.H1("Genomic Risk Landscape"), html.P("SHAP-driven gene importance: genes de risco vs protetores.")], className="page-header"),
        html.Div([
            kpi_card("Genes de Risco", n_risk, "SHAP > 0", "red"),
            kpi_card("Genes Protetores", n_prot, "SHAP < 0", "green"),
            kpi_card("Total Features", model_meta.get('n_features', 'N/A'), "No modelo", "blue"),
        ], className="kpi-grid"),
        html.Div([
            html.Div([html.Div("SHAP Feature Importance — Top 25", className="card-title"),
                       dcc.Graph(figure=fig_shap)], className="card full-width"),
        ], className="chart-grid"),
        html.Div([
            html.Div("Gene Risk Rankings", className="card-title"),
            dash_table.DataTable(data=table_data, page_size=15,
                style_header={'backgroundColor':'#1e293b','color':'#94a3b8','fontWeight':'600','fontSize':'12px','border':'none'},
                style_cell={'backgroundColor':'#111827','color':'#f1f5f9','border':'1px solid #1e293b','fontSize':'13px','padding':'8px 12px'},
                style_data_conditional=[
                    {'if':{'filter_query':'{direction} = "RISCO"','column_id':'direction'},'color':COLORS["risk"],'fontWeight':'bold'},
                    {'if':{'filter_query':'{direction} = "PROTETOR"','column_id':'direction'},'color':COLORS["safe"],'fontWeight':'bold'},
                ])
        ], className="card"),
    ], className="fade-in")

# ============================================================
# PAGE: MUTATIONS
# ============================================================
def page_mutations():
    fig_mut = go.Figure()
    if not df_mutations.empty and 'gene_symbol' in df_mutations.columns:
        top_genes = df_mutations['gene_symbol'].value_counts().head(20)
        fig_mut.add_trace(go.Bar(x=top_genes.values, y=top_genes.index, orientation='h',
                                  marker=dict(color=top_genes.values, colorscale='Reds')))
    fig_mut.update_layout(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=500, xaxis_title="Nº Mutações", yaxis=dict(autorange="reversed"),
                          margin=dict(t=30,b=50,l=120,r=30))

    fig_type = go.Figure()
    if not df_mutations.empty and 'consequence_type' in df_mutations.columns:
        types = df_mutations['consequence_type'].value_counts().head(10)
        fig_type.add_trace(go.Pie(labels=types.index, values=types.values, hole=0.5,
                                   marker_colors=px.colors.qualitative.Set3))
    fig_type.update_layout(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           height=400, margin=dict(t=30,b=30,l=30,r=30))

    n_muts = len(df_mutations)
    n_genes = df_mutations['gene_symbol'].nunique() if not df_mutations.empty and 'gene_symbol' in df_mutations.columns else 0

    return html.Div([
        html.Div([html.H1("Perfil Mutacional"), html.P("Mutações somáticas (SSMs) extraídas do GDC.")], className="page-header"),
        html.Div([kpi_card("Total Mutações", f"{n_muts:,}", "", "red"), kpi_card("Genes Mutados", n_genes, "", "purple")], className="kpi-grid"),
        html.Div([
            html.Div([html.Div("Top 20 Genes Mais Mutados", className="card-title"), dcc.Graph(figure=fig_mut)], className="card"),
            html.Div([html.Div("Tipos de Consequência", className="card-title"), dcc.Graph(figure=fig_type)], className="card"),
        ], className="chart-grid"),
    ], className="fade-in")

# ============================================================
# PAGE: TREATMENT
# ============================================================
def page_treatment():
    fig_treat = go.Figure()
    if not df_clinical.empty and 'tipos_tratamento' in df_clinical.columns:
        treat_counts = {}
        treat_survival = {}
        for _, row in df_clinical.iterrows():
            for t in str(row.get('tipos_tratamento','')).split(' | '):
                t = t.strip()
                if t and t != 'Não Informado':
                    treat_counts[t] = treat_counts.get(t, 0) + 1
                    treat_survival.setdefault(t, []).append(row['dias_sobrevivencia'])
        top_treats = sorted(treat_counts.items(), key=lambda x: x[1], reverse=True)[:12]
        names = [t[0][:30] for t in top_treats]
        medians = [np.median(treat_survival[t[0]]) for t in top_treats]
        fig_treat.add_trace(go.Bar(x=medians, y=names, orientation='h',
                                    marker=dict(color=medians, colorscale='Viridis'),
                                    text=[f"{m:.0f}d" for m in medians], textposition='outside'))
    fig_treat.update_layout(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            height=500, xaxis_title="Mediana Sobrevivência (dias)", yaxis=dict(autorange="reversed"),
                            margin=dict(t=30,b=50,l=250,r=80))

    return html.Div([
        html.Div([html.H1("Treatment Response"), html.P("Análise de resposta por tipo de tratamento.")], className="page-header"),
        html.Div([
            html.Div([html.Div("Mediana de Sobrevivência por Tratamento", className="card-title"),
                       dcc.Graph(figure=fig_treat)], className="card full-width"),
        ], className="chart-grid"),
    ], className="fade-in")

# ============================================================
# PAGE: NETWORK
# ============================================================
def page_network():
    nodes_path = os.path.join(BASE, "data/processed/gbm_genes_nodes.csv")
    edges_path = os.path.join(BASE, "data/processed/gbm_network_edges.csv")
    elements = []
    if os.path.exists(nodes_path) and os.path.exists(edges_path):
        df_n = pd.read_csv(nodes_path).head(50)
        df_e = pd.read_csv(edges_path)
        node_ids = set(df_n.iloc[:,0].astype(str).str.split('.').str[0])
        for nid in list(node_ids)[:50]:
            elements.append({'data': {'id': nid, 'label': nid[:15]}, 'classes': 'gene-node'})
        df_e['src'] = df_e['gene_source'].astype(str).str.split('.').str[0]
        df_e['tgt'] = df_e['gene_target'].astype(str).str.split('.').str[0]
        df_e_filt = df_e[(df_e['src'].isin(node_ids)) & (df_e['tgt'].isin(node_ids))].head(200)
        for _, r in df_e_filt.iterrows():
            elements.append({'data': {'source': r['src'], 'target': r['tgt'], 'weight': abs(r['weight'])}})

    cyto_graph = cyto.Cytoscape(
        id='gene-network', elements=elements, layout={'name': 'cose', 'animate': False},
        style={'width': '100%', 'height': '600px', 'backgroundColor': '#111827'},
        stylesheet=[
            {'selector': 'node', 'style': {'label': 'data(label)', 'background-color': '#3b82f6',
                'color': '#f1f5f9', 'font-size': '10px', 'width': 30, 'height': 30,
                'border-width': 2, 'border-color': '#1e40af'}},
            {'selector': 'edge', 'style': {'line-color': 'rgba(59,130,246,0.3)', 'width': 1,
                'curve-style': 'bezier'}},
        ]
    ) if elements else html.Div("⚠️ Dados de rede não encontrados.", style={"color": "#94a3b8", "padding": "40px"})

    return html.Div([
        html.Div([html.H1("Gene Network Explorer"), html.P("Rede de co-expressão gênica (top 50 nós).")], className="page-header"),
        html.Div([html.Div("Rede de Co-Expressão", className="card-title"), cyto_graph], className="card"),
    ], className="fade-in")

# ============================================================
# PAGE: PRESCRIPTIVE
# ============================================================
def page_prescriptive():
    risk_genes = prescriptive.get('gene_alerts', {}).get('risk_genes', [])
    prot_genes = prescriptive.get('gene_alerts', {}).get('protective_genes', [])

    risk_list = html.Div([
        html.H3("🔴 Genes de Alto Risco", style={"color": COLORS["risk"], "marginBottom": "12px", "fontSize": "16px"}),
        *[html.Div([
            html.Span(g['feature'], style={"fontWeight":"600"}),
            html.Span(f" SHAP: {g['mean_shap']:+.4f}", style={"color": COLORS["risk"], "marginLeft":"8px", "fontSize":"13px"})
        ], style={"padding":"6px 0", "borderBottom":f"1px solid {COLORS['bg']}"}) for g in risk_genes[:10]]
    ]) if risk_genes else html.P("Execute explainability.py primeiro.", style={"color":"#64748b"})

    prot_list = html.Div([
        html.H3("🟢 Genes Protetores", style={"color": COLORS["safe"], "marginBottom": "12px", "fontSize": "16px"}),
        *[html.Div([
            html.Span(g['feature'], style={"fontWeight":"600"}),
            html.Span(f" SHAP: {g['mean_shap']:+.4f}", style={"color": COLORS["safe"], "marginLeft":"8px", "fontSize":"13px"})
        ], style={"padding":"6px 0", "borderBottom":f"1px solid {COLORS['bg']}"}) for g in prot_genes[:10]]
    ]) if prot_genes else html.P("Execute explainability.py primeiro.", style={"color":"#64748b"})

    return html.Div([
        html.Div([html.H1("Prescriptive Analytics"), html.P("Recomendações clínicas baseadas em dados.")], className="page-header"),
        html.Div([
            kpi_card("Genes de Risco", len(risk_genes), "Alto impacto SHAP", "red"),
            kpi_card("Genes Protetores", len(prot_genes), "Efeito protetor", "green"),
            kpi_card("Pacientes Analisados", prescriptive.get('total_patients', 'N/A'), "", "blue"),
        ], className="kpi-grid"),
        html.Div([
            html.Div([html.Div("Alertas Genômicos", className="card-title"), risk_list], className="card"),
            html.Div([html.Div("Fatores Protetores", className="card-title"), prot_list], className="card"),
        ], className="chart-grid"),
    ], className="fade-in")

# ============================================================
# LAYOUT & ROUTING
# ============================================================
app.layout = html.Div([
    dcc.Store(id='current-page', data='overview'),
    sidebar,
    html.Div(id='page-content', className="main-content"),
], className="app-container")

PAGES = {
    "overview": page_overview, "survival": page_survival, "genomic": page_genomic,
    "mutations": page_mutations, "treatment": page_treatment, "network": page_network,
    "prescriptive": page_prescriptive,
}

@callback(Output('page-content', 'children'), Output('current-page', 'data'),
          *[Input(f'nav-{p}', 'n_clicks') for p in PAGES], State('current-page', 'data'))
def navigate(*args):
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == '.':
        return page_overview(), 'overview'
    page_id = ctx.triggered[0]['prop_id'].split('.')[0].replace('nav-', '')
    if page_id in PAGES:
        return PAGES[page_id](), page_id
    return page_overview(), 'overview'

if __name__ == "__main__":
    print("🚀 Dashboard: http://localhost:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)
