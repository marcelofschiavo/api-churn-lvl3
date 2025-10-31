"""
===================================================================
PROJETO 1 (NÍVEL 4) - SCRIPT 3: app.py (API Versão BigQuery)
===================================================================
"""
import uvicorn
import joblib
import pandas as pd
import os
import json # <--- NOVO: Para ler o JSON injetado
from fastapi import FastAPI
from pydantic import BaseModel 
from google.cloud import bigquery # <--- NOVO: Para o BigQuery

# --- 🧠 Configuração Nível 4 (O BQ e Caminhos) ---
PROJECT_ID_CORRETO = "churn-api-476801"
BQ_DATASET = "rh_data" 

# 🧠 Explicação: Agora, o app.py espera encontrar a chave injetada na raiz
KEY_PATH = "bigquery_key.json" 
MODEL_PATH = "models/best_churn_pipeline.pkl" # Corrigido para caminho relativo

# --- 1. Criando a Instância da API ---
app = FastAPI(
    # Mude o título para refletir a arquitetura Nível 4 (BigQuery)
    title="API de Predição de Churn (Arquitetura Nível 4 | MLOps BigQuery)", 
    description="Modelo treinado no Google BigQuery, servido via FastAPI."
)


# --- 2. Carregando o Modelo e Autenticando o BQ ---
# 🧠 Explicação: O 'os.environ' deve apontar para o arquivo que o Docker injetou.
print("--- Carregando Modelo e Autenticando BQ ---")

try:
    # 1. Autenticação BQ (Obrigatório antes do joblib.load)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
    
    # 2. Carregar o Modelo
    model = joblib.load(MODEL_PATH)
    print("Modelo e BQ autenticados com sucesso.")

except FileNotFoundError:
    print(f"ERRO CRÍTICO: Não encontrou o modelo ou a chave em {MODEL_PATH} ou {KEY_PATH}")
    model = None
except Exception as e:
     print(f"ERRO DE INICIALIZAÇÃO: {e}")
     model = None


# --- 3. Definindo o "Contrato" de Entrada (Schema Pydantic) ---
# 🧠 Explicação: Precisamos dizer à API quais dados ela DEVE esperar.
# Isso é um "contrato" (Schema). Se o usuário mandar dados faltando
# ou com o tipo errado (ex: 'salario' como texto), o FastAPI
# automaticamente retorna um erro 422 (Dados Inválidos).
#
# As colunas aqui DEVEM ser as mesmas que o Script 01 gerou
# *antes* do One-Hot Encoding.
class FuncionarioFeatures(BaseModel):
    salario_mensal: float
    tempo_empresa_dias: int
    dias_desde_ultimo_login: int
    media_tempo_logado_min: float
    total_chamados_suporte: int
    departamento: str # Ex: "Vendas", "TI", "Marketing"
    
    # 🧠 Exemplo de como o usuário mandaria no JSON (o "pedido"):
    # {
    #   "salario_mensal": 5000,
    #   "tempo_empresa_dias": 700,
    #   "dias_desde_ultimo_login": 10,
    #   "media_tempo_logado_min": 55.5,
    #   "total_chamados_suporte": 1,
    #   "departamento": "Vendas"
    # }


# --- 4. Criando o "Endpoint" de Predição (O "Garçom") ---
# 🧠 Explicação: @app.post("/predict") diz:
# "Ei, FastAPI! Se alguém fizer um pedido 'POST' para a URL '/predict',
# execute esta função."
@app.post("/predict")
def predict_churn(features: FuncionarioFeatures):
    """
    Recebe os dados de um funcionário e retorna a probabilidade de churn.
    """
    if model is None:
        return {"erro": "Modelo não foi carregado. API inoperante."}

    # 1. Converter os dados do JSON (Pydantic) para um DataFrame
    # 🧠 Explicação: O modelo foi treinado em um DataFrame, então ele espera um.
    # Esta é a parte mais "frágil" e importante do Lvl 3.
    # Precisamos replicar *exatamente* as colunas que o modelo espera.
    
    dados_entrada = {
        'salario_mensal': [features.salario_mensal],
        'tempo_empresa_dias': [features.tempo_empresa_dias],
        'dias_desde_ultimo_login': [features.dias_desde_ultimo_login],
        'media_tempo_logado_min': [features.media_tempo_logado_min],
        'total_chamados_suporte': [features.total_chamados_suporte],
        
        # 🧠 Explicação (One-Hot Encoding Manual):
        # Nosso pipeline salvo (o .pkl) *NÃO* sabe converter "Vendas" em 0/1.
        # Nós fizemos isso *antes* no Script 02.
        # A API tem que fazer a mesma coisa!
        'departamento_TI': [1 if features.departamento == 'TI' else 0],
        'departamento_Vendas': [1 if features.departamento == 'Vendas' else 0]
        # 'Marketing' é o drop_first=True (quando TI=0 e Vendas=0)
    }
    
    # Garantir a ordem das colunas (tem que ser IDÊNTICA ao treino)
    colunas_do_modelo = [
        'salario_mensal', 'tempo_empresa_dias', 'dias_desde_ultimo_login',
        'media_tempo_logado_min', 'total_chamados_suporte', 
        'departamento_TI', 'departamento_Vendas'
    ]
    
    try:
        df_para_prever = pd.DataFrame(dados_entrada, columns=colunas_do_modelo)
    except Exception as e:
        return {"erro": f"Erro ao criar DataFrame: {e}"}

    # 2. Fazer a Predição
    # 🧠 Explicação: O pipeline (.pkl) executa o Scaler, o SMOTE (só no treino)
    # e o 'predict_proba' do modelo.
    # Queremos a *probabilidade* (predict_proba), não só 0 ou 1.
    try:
        pred_proba = model.predict_proba(df_para_prever)
        
        # A saída de predict_proba é [prob_classe_0, prob_classe_1]
        probabilidade_de_sair = pred_proba[0][1] # Pegamos a prob de "1" (Sair)
        predicao = bool(probabilidade_de_sair > 0.5) # Converte para True/False

    except Exception as e:
        return {"erro": f"Erro ao fazer predição: {e}"}

    # 3. Retornar o resultado para o usuário (em JSON)
    return {
        "probabilidade_de_churn": float(probabilidade_de_sair),
        "predicao_churn": predicao,
        "dados_recebidos": features
    }


# --- 5. (Opcional, mas Recomendado) Endpoint de "Saúde" ---
@app.get("/health", tags=["Monitoring"])
def health_check():
    # 🧠 Explicação: O monitoramento (Lvl 3) precisa saber se
    # a API está "viva". Esse endpoint é o "monitor cardíaco".
    if model is not None:
        return {"status": "ok", "message": "API de Churn (Lvl 3) está no ar e modelo carregado!"}
    else:
        return {"status": "error", "message": "API no ar, mas MODELO NÃO ENCONTRADO!"}


# --- 6. Rodando a API (Quando executamos 'python src/03_api.py') ---
if __name__ == "__main__":
    print("--- Rodando a API localmente (modo de desenvolvimento) ---")
    print("Acesse http://127.0.0.1:8000/docs para testar.")
    # 🧠 Explicação: 'reload=True' é ótimo para dev.
    # Se você salvar o script, a API reinicia sozinha.
    uvicorn.run("app:app", host="127.0.0.1", port=7860, reload=True)