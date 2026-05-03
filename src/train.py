"""
train.py — Pipeline Profissional de Machine Learning para GBM
v2.0 — Com LightGBM, CatBoost, Stacking Ensemble, Optuna, métricas avançadas

Modelos: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, Stacking
Tuning: Optuna (Bayesian TPE)
Métricas: ROC-AUC, F1, MCC, Brier Score, Precision-Recall AUC
"""
import os, json, time
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc,
    classification_report, confusion_matrix, matthews_corrcoef, brier_score_loss,
    log_loss
)
from sklearn.feature_selection import SelectFromModel
from dotenv import load_dotenv
load_dotenv()

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import boto3


# ============================================================
# DATA LOADING & MERGING
# ============================================================
def load_and_merge_data(clinical_path, expression_path, nodes_path):
    """Carrega dados clínicos, filtra genes VIP e funde."""
    print("📂 Carregando gabarito clínico...")
    df_clinical = pd.read_csv(clinical_path)
    df_clinical['paciente_id'] = df_clinical['paciente_id'].astype(str).str.strip()

    print("🧠 Lendo lista de Genes de Alta Confiança (Neo4j)...")
    df_nodes = pd.read_csv(nodes_path)
    col = 'gene_id' if 'gene_id' in df_nodes.columns else df_nodes.columns[0]
    genes_validos = set(df_nodes[col].astype(str).str.split('.').str[0].str.strip())
    print(f"🎯 {len(genes_validos)} genes VIPs")

    print("⏳ Lendo matriz em chunks...")
    chunks = []
    for chunk in pd.read_csv(expression_path, index_col=0, chunksize=5000):
        chunk.index = chunk.index.astype(str).str.split('.').str[0].str.strip()
        c = chunk[chunk.index.isin(genes_validos)].copy()
        if not c.empty:
            c = c[~c.index.duplicated(keep='first')]
            chunks.append(c)

    if not chunks:
        print("❌ Nenhum gene encontrado na matriz.")
        return pd.DataFrame()

    df_expr = pd.concat(chunks).T
    df_expr.columns = df_expr.columns.astype(str)
    df_expr.index = df_expr.index.astype(str).str.strip().str[:12]

    df_merged = pd.merge(df_clinical, df_expr, left_on='paciente_id', right_index=True, how='inner')
    print(f"✅ Fusão: {df_merged.shape[0]} pacientes × {df_expr.shape[1]} genes")
    return df_merged


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_data(df):
    """Pré-processa o dataset com OrdinalEncoder, Imputer mediana e Scaler."""
    print("⚙️  Pré-processando...")
    colunas_ignoradas = ['paciente_id','status_vital','dias_sobrevivencia','classe_risco','is_censored']
    y = df['classe_risco'].map({'Alto Risco': 1, 'Baixo Risco': 0})
    X = df.drop(columns=[c for c in colunas_ignoradas if c in df.columns])

    # OHE tratamentos
    if 'tipos_tratamento' in X.columns:
        tratamentos = X['tipos_tratamento'].str.get_dummies(sep=' | ')
        X = pd.concat([X, tratamentos], axis=1).drop(columns=['tipos_tratamento'])

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    encoder_map = {}
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols].astype(str))
        encoder_map = {'ordinal_encoder': enc, 'cat_cols': cat_cols,
                       'categories': {col: list(enc.categories_[i]) for i, col in enumerate(cat_cols)}}

    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    # Sanitize column names for LightGBM (no special JSON chars)
    import re
    clean_cols = [re.sub(r'[^a-zA-Z0-9_]', '_', str(c)) for c in X_imp.columns]
    X_imp.columns = clean_cols

    print(f"✅ Shape final: {X_imp.shape}")
    return X_imp, y, encoder_map, imputer, None


# ============================================================
# MODEL EVALUATION
# ============================================================
def evaluate_model(model, X_test, y_test, model_name="Modelo"):
    """Avalia modelo com métricas completas."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    metrics = {'accuracy': acc, 'f1_weighted': f1, 'mcc': mcc}

    if y_proba is not None:
        roc = roc_auc_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
        ll = log_loss(y_test, y_proba)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(rec, prec)
        metrics.update({'roc_auc': roc, 'pr_auc': pr_auc, 'brier': brier, 'log_loss': ll})
    else:
        roc = 'N/A'

    print(f"\n📊 {model_name}:")
    print(f"   Acurácia : {acc*100:.2f}%  |  ROC-AUC: {roc if isinstance(roc,str) else f'{roc:.4f}'}")
    print(f"   F1       : {f1:.4f}    |  MCC    : {mcc:.4f}")
    if y_proba is not None:
        print(f"   PR-AUC   : {pr_auc:.4f}  |  Brier  : {brier:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Baixo Risco','Alto Risco']))
    return metrics


# ============================================================
# OPTUNA HYPERPARAMETER TUNING
# ============================================================
def optuna_tune_xgboost(X_train, y_train, cv, n_trials=50):
    """Tuning Bayesiano do XGBoost com Optuna."""
    if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE:
        return {}
    print("\n🔧 Optuna: Tuning XGBoost...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 3),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5, log=True),
        }
        model = XGBClassifier(**params, scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                              random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1)
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            val = scores.mean()
            return val if not np.isnan(val) else 0.5
        except Exception:
            return 0.5

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    try:
        print(f"   🏆 Melhor ROC-AUC: {study.best_value:.4f}")
        return study.best_params
    except ValueError:
        print("   ⚠️  Optuna não encontrou trials válidos. Usando defaults.")
        return {}

def optuna_tune_lgbm(X_train, y_train, cv, n_trials=50):
    """Tuning Bayesiano do LightGBM com Optuna."""
    if not OPTUNA_AVAILABLE or not LGBM_AVAILABLE:
        return {}
    print("\n🔧 Optuna: Tuning LightGBM...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5, log=True),
        }
        model = LGBMClassifier(**params, is_unbalance=True, random_state=42, verbosity=-1, n_jobs=-1)
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            val = scores.mean()
            return val if not np.isnan(val) else 0.5
        except Exception:
            return 0.5

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    try:
        print(f"   🏆 Melhor ROC-AUC: {study.best_value:.4f}")
        return study.best_params
    except ValueError:
        print("   ⚠️  Optuna não encontrou trials válidos. Usando defaults.")
        return {}


# ============================================================
# TRAINING & EVALUATION
# ============================================================
def train_and_evaluate(X, y, n_optuna_trials=50):
    """Treina, tuna e avalia múltiplos modelos."""
    print("\n🔀 Split treino (70%) / teste (30%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---- Optuna tuning ----
    xgb_params = optuna_tune_xgboost(X_train, y_train, cv, n_trials=n_optuna_trials) if OPTUNA_AVAILABLE else {}
    lgbm_params = optuna_tune_lgbm(X_train, y_train, cv, n_trials=n_optuna_trials) if OPTUNA_AVAILABLE else {}

    # ---- Candidatos ----
    candidatos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_split=10, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42),
    }

    if XGBOOST_AVAILABLE:
        xgb_final_params = {
            'n_estimators': xgb_params.get('n_estimators', 200),
            'max_depth': xgb_params.get('max_depth', 4),
            'learning_rate': xgb_params.get('learning_rate', 0.05),
            'subsample': xgb_params.get('subsample', 0.8),
            'colsample_bytree': xgb_params.get('colsample_bytree', 0.8),
            'min_child_weight': xgb_params.get('min_child_weight', 3),
            'gamma': xgb_params.get('gamma', 0),
            'reg_alpha': xgb_params.get('reg_alpha', 0.01),
            'reg_lambda': xgb_params.get('reg_lambda', 1),
            'scale_pos_weight': (y==0).sum()/(y==1).sum(),
            'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0, 'n_jobs': -1
        }
        candidatos["XGBoost (Optuna)"] = XGBClassifier(**xgb_final_params)

    if LGBM_AVAILABLE:
        lgbm_final = {
            'n_estimators': lgbm_params.get('n_estimators', 200),
            'max_depth': lgbm_params.get('max_depth', 5),
            'learning_rate': lgbm_params.get('learning_rate', 0.05),
            'subsample': lgbm_params.get('subsample', 0.8),
            'colsample_bytree': lgbm_params.get('colsample_bytree', 0.8),
            'num_leaves': lgbm_params.get('num_leaves', 50),
            'min_child_samples': lgbm_params.get('min_child_samples', 10),
            'reg_alpha': lgbm_params.get('reg_alpha', 0.01),
            'reg_lambda': lgbm_params.get('reg_lambda', 1),
            'is_unbalance': True, 'random_state': 42, 'verbosity': -1, 'n_jobs': -1
        }
        candidatos["LightGBM (Optuna)"] = LGBMClassifier(**lgbm_final)

    if CATBOOST_AVAILABLE:
        candidatos["CatBoost"] = CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.05, auto_class_weights='Balanced',
            random_seed=42, verbose=0)

    # ---- Cross-Validation ----
    resultados_cv = {}
    modelos_treinados = {}
    print("\n🧪 Cross-Validation (5-fold)...")
    for nome, modelo in candidatos.items():
        print(f"\n   ⏳ {nome}...")
        auc_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        acc_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        f1_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        print(f"      ROC-AUC : {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
        print(f"      Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
        print(f"      F1      : {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
        resultados_cv[nome] = {'roc_auc': auc_scores.mean(), 'accuracy': acc_scores.mean(), 'f1': f1_scores.mean()}
        modelo.fit(X_train, y_train)
        modelos_treinados[nome] = modelo

    # ---- Stacking Ensemble ----
    base_models = [(n, m) for n, m in modelos_treinados.items() if n != "Gradient Boosting"]
    if len(base_models) >= 2:
        print("\n🏗️  Construindo Stacking Ensemble...")
        stacking = StackingClassifier(
            estimators=base_models[:4],
            final_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            cv=5, n_jobs=-1, passthrough=False
        )
        stk_auc = cross_val_score(stacking, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        print(f"      Stacking ROC-AUC: {stk_auc.mean():.4f} ± {stk_auc.std():.4f}")
        stacking.fit(X_train, y_train)
        candidatos["Stacking Ensemble"] = stacking
        modelos_treinados["Stacking Ensemble"] = stacking
        resultados_cv["Stacking Ensemble"] = {'roc_auc': stk_auc.mean()}

    # ---- Melhor modelo ----
    melhor_nome = max(resultados_cv, key=lambda k: resultados_cv[k]['roc_auc'])
    print(f"\n🏆 Melhor: {melhor_nome} (ROC-AUC CV: {resultados_cv[melhor_nome]['roc_auc']:.4f})")

    # ---- Hold-out evaluation ----
    print("\n" + "="*60)
    all_metrics = {}
    for nome, modelo in modelos_treinados.items():
        metrics = evaluate_model(modelo, X_test, y_test, model_name=nome)
        all_metrics[nome] = metrics

    # ---- Feature importance ----
    melhor = modelos_treinados[melhor_nome]
    if hasattr(melhor, 'feature_importances_'):
        imp = pd.DataFrame({'Feature': X.columns, 'Importancia': melhor.feature_importances_}
                           ).sort_values('Importancia', ascending=False)
        print(f"\n🏅 Top 20 Features ({melhor_nome}):")
        print(imp.head(20).to_string(index=False))

    return melhor, melhor_nome, all_metrics, resultados_cv


# ============================================================
# S3 UPLOAD
# ============================================================
def upload_to_s3(file_name, bucket_name):
    print(f"☁️  Upload {file_name} → S3 ({bucket_name})...")
    try:
        boto3.client('s3').upload_file(file_name, bucket_name, os.path.basename(file_name))
        print("✅ Upload OK!")
    except Exception as e:
        print(f"❌ Erro S3: {e}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    CLINICAL_PATH   = "data/processed/clinical_gbm_completo.csv"
    EXPRESSION_PATH = "data/raw/matriz_gbm_bruta.csv"
    MODEL_PATH      = "models/modelo_sobrevivencia_gbm_v2.pkl"
    META_PATH       = "models/modelo_metadata_v2.json"
    COMPARISON_PATH = "models/model_comparison_report.json"
    NODES_PATH      = "data/processed/gbm_genes_nodes.csv"
    N_OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "50"))
    S3_UPLOAD       = os.getenv("S3_UPLOAD_ENABLED", "false").lower() == "true"
    BUCKET          = os.getenv("S3_BUCKET_NAME", "gbm-ml-models-bucket")

    t0 = time.time()

    # 1. Load & merge
    df_completo = load_and_merge_data(CLINICAL_PATH, EXPRESSION_PATH, NODES_PATH)
    if df_completo.empty:
        print("⚠️  Sem dados. Verifique CSVs."); exit()

    # 2. Preprocess
    X, y, encoder_map, imputer, scaler = preprocess_data(df_completo)
    print(f"\n📊 Dataset: {X.shape[0]} amostras × {X.shape[1]} features")
    print(f"   Alto Risco : {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    print(f"   Baixo Risco: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")

    # 3. Train & evaluate
    modelo, nome, all_metrics, cv_results = train_and_evaluate(X, y, n_optuna_trials=N_OPTUNA_TRIALS)

    # 4. Save model package
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    pkg = {
        'model': modelo, 'model_name': nome,
        'imputer': imputer, 'scaler': scaler,
        'encoder_map': encoder_map, 'feature_names': list(X.columns),
    }
    joblib.dump(pkg, MODEL_PATH)
    print(f"\n💾 Modelo v2 salvo: {MODEL_PATH}")

    # 5. Save metadata
    meta = {
        'model_name': nome, 'n_samples': int(X.shape[0]), 'n_features': int(X.shape[1]),
        'class_distribution': {'Alto Risco': int((y==1).sum()), 'Baixo Risco': int((y==0).sum())},
        'cv_results': {k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in cv_results.items()},
        'holdout_metrics': {k: {mk: round(mv, 4) if isinstance(mv, float) else mv for mk, mv in v.items()}
                           for k, v in all_metrics.items()},
        'training_time_seconds': round(time.time()-t0, 1),
        'feature_names': list(X.columns)[:50],  # top 50 para legibilidade
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"📋 Metadata: {META_PATH}")

    # 6. Save comparison report
    with open(COMPARISON_PATH, 'w') as f:
        json.dump({'cv_results': cv_results, 'holdout_metrics': all_metrics}, f, indent=2, default=str)
    print(f"📊 Comparação: {COMPARISON_PATH}")

    # 7. S3
    if S3_UPLOAD:
        upload_to_s3(MODEL_PATH, BUCKET)
    else:
        print("⏭️  S3 desativado (S3_UPLOAD_ENABLED=false)")

    print(f"\n⏱️  Tempo total: {time.time()-t0:.1f}s")