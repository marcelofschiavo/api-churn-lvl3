"""
===================================================================
PROJETO 1 (NÍVEL 3) - SCRIPT 3: 03_api.py

OBJETIVO: (Deploy / MLOps)
1.  (Load) Carregar a pipeline '.pkl' salva pelo Script 02.
2.  (Serve) Criar uma API FastAPI para "servir" o modelo.
3.  (Endpoint) Criar um endpoint '/predict' que recebe dados
    de um funcionário (em JSON) e retorna o risco de churn.

COMO RODAR (no terminal, com o .venv ativado):
$ python src/03_api.py
(E depois acesse http://127.0.0.1:8000/docs no seu navegador)
===================================================================
"""

import uvicorn  # O servidor que "liga" a API
import joblib   # Para carregar nosso modelo .pkl
import pandas as pd
import os
from fastapi import FastAPI
from pydantic import BaseModel # Para definir os "contratos" (schemas) de entrada

# --- 🧠 Explicação Didática: Caminhos (Paths) ---
# Encontrando o caminho para o nosso modelo salvo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_churn_pipeline.pkl')


# --- 1. Criando a Instância da API ---
# 'app' é a instância principal da nossa API
print("--- Iniciando a API do Modelo de Churn (Lvl 3) ---")
app = FastAPI(
    title="API de Predição de Churn (Lvl 3)",
    description="Serve um modelo RF+SMOTE para prever o risco de saída de funcionários."
)


# --- 2. Carregando o Modelo (O "Produto" do Lvl 2) ---
# 🧠 Explicação: Carregamos o modelo UMA VEZ quando a API liga.
# Ele fica "vivo" na memória, pronto para ser usado.
print(f"Carregando modelo de: {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
    print("Modelo carregado com sucesso na memória.")
except FileNotFoundError:
    print(f"ERRO: Modelo '{MODEL_PATH}' não encontrado!")
    print("Por favor, rode 'python src/02_train.py' primeiro.")
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
    uvicorn.run("03_api:app", host="127.0.0.1", port=7860, reload=True)