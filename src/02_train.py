"""
===================================================================
PROJETO 1 (NÍVEL 2.5) - SCRIPT 2: 02_train.py (Versão Completa)

OBJETIVO: (Treinamento e Otimização)
- Corrigido para incluir definições de path e todos os imports.
===================================================================
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib 
import os
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    recall_score, 
    precision_score, 
    accuracy_score
)
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# --- 🧠 DEFINIÇÃO DOS CAMINHOS (Paths) ---
# (Esta era a parte que estava faltando)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'master_dataframe_limpo.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_churn_pipeline.pkl')


def treinar_modelo(data_path, model_path):
    """
    Função principal que orquestra todo o treinamento do modelo.
    """
    print("--- INICIANDO PIPELINE DE TREINAMENTO (NÍVEL 2.5 - Corrigido) ---")
    
    # --- 1. Setup do MLflow ---
    mlflow.set_experiment("Projeto Churn Lvl 2")
    mlflow.start_run(run_name="Tuning_RF_com_SMOTE")
    print("Experimento MLflow 'Projeto Churn Lvl 2' iniciado.")
    
    # --- 2. LOAD & PREPARE ---
    print(f"Carregando dados limpos de: {data_path}")
    try:
        df_master = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {data_path}")
        print("Por favor, rode 'python src/01_data_preprocessing.py' primeiro.")
        return

    df_master = pd.get_dummies(df_master, columns=['departamento'], drop_first=True, dtype=int)
    
    y = df_master['pediu_demissao']
    X = df_master.drop(['id_funcionario', 'pediu_demissao'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=42, 
        stratify=y
    )
    print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

    # --- 3. PIPELINE ---
    pipeline_rf = ImblearnPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    # --- 4. TUNE (GridSearchCV) ---
    print("Iniciando o Tuning (GridSearchCV)...")
    param_grid = {
        'model__n_estimators': [100, 150],
        'model__max_depth': [5, 10],
    }
    grid_search = GridSearchCV(
        estimator=pipeline_rf,
        param_grid=param_grid,
        cv=3,
        scoring='recall',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # --- 5. EVALUATE ---
    print("Tuning concluído. Avaliando o melhor modelo...")
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Melhores Parâmetros encontrados: {best_params}")

    y_pred = best_pipeline.predict(X_test)
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\n--- Resultados no Set de Teste (Nível 2.5) ---")
    print(f"Accuracy: {acc:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"**Recall: {recall:.2%}**")
    print(f"ROC AUC: {roc_auc:.2%}")
    
    # --- 6. TRACK (Registrando no MLflow) ---
    print("Registrando resultados no MLflow (com Assinatura)...")

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Criar o "Exemplo de Entrada" (input_example)
    input_example = X_train.head(5)
    
    # Registrar o modelo (pipeline inteira) no MLflow com a assinatura
    mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        artifact_path="model_pipeline",  # O nome da "pasta" do modelo
        input_example=input_example,     # O "contrato" (assinatura)
        registered_model_name="churn_rf_smote_v1" # O "nome oficial"
    )

    # --- 7. SAVE (Salvando o "Produto" do Nível 2) ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    
    print(f"\n--- Modelo campeão salvo em: '{model_path}' ---")
    mlflow.end_run()
    print("--- PIPELINE DE TREINAMENTO CONCLUÍDA! ---")


if __name__ == "__main__":
    # Agora DATA_PATH e MODEL_PATH existem e podem ser passados
    treinar_modelo(DATA_PATH, MODEL_PATH)