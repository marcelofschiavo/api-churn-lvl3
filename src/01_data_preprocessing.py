"""
===================================================================
PROJETO 1 (NÍVEL 2) - SCRIPT 1: 01_data_preprocessing.py

OBJETIVO: (ETL - Extract, Transform, Load)
1.  (Extract) Carregar os 3 CSVs bagunçados da pasta /data.
2.  (Transform) Limpar, agregar e juntar os dados.
3.  (Load) Salvar um único CSV limpo e pronto para o ML.

COMO RODAR (no terminal, com o .venv ativado):
$ python src/01_data_preprocessing.py
===================================================================
"""

import pandas as pd
import numpy as np
import datetime
import os

# --- 🧠 Explicação Didática: Caminhos (Paths) ---
# Usar 'os.path.join' é uma boa prática (Lvl 2).
# Ele funciona no Windows, Mac ou Linux (evita usar '/' ou '\' errado).
# '..' sobe um nível (da pasta 'src' para a pasta 'projeto_churn_lvl2')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data') # Vamos salvar no mesmo lugar

PATH_RH = os.path.join(DATA_DIR, 'funcionarios.csv')
PATH_LOGS = os.path.join(DATA_DIR, 'logs_acesso.csv')
PATH_SUPORTE = os.path.join(DATA_DIR, 'chamados_suporte.csv')
PATH_OUTPUT = os.path.join(OUTPUT_DIR, 'master_dataframe_limpo.csv')


def processar_dados_rh(path):
    """
    Carrega e limpa os dados principais do RH.
    - Trata salários nulos (NaN)
    - Cria a variável alvo (target) 'pediu_demissao'
    - Calcula o 'tempo_empresa_dias'
    """
    print(f"Processando RH: {path}")
    df_rh = pd.read_csv(path)

    # 🧠 Explicação: Tratar nulos (Lvl 1)
    # Preencher salários faltantes com a *mediana* do depto (mais robusto que a média)
    mediana_salario_geral = df_rh['salario_mensal'].median()
    df_rh['salario_mensal'] = df_rh['salario_mensal'].fillna(mediana_salario_geral)

    # 🧠 Explicação: Criar a variável Target (nosso 'y')
    # Convertendo texto ('status') em número (0 ou 1)
    df_rh['pediu_demissao'] = np.where(df_rh['status'] == 'Inativo (Saiu)', 1, 0)

    # 🧠 Explicação: Feature Engineering (Nível 1)
    # Modelos de ML entendem melhor "365 dias" do que "10/10/2023"
    df_rh['data_contratacao'] = pd.to_datetime(df_rh['data_contratacao'])
    hoje = datetime.datetime.today()
    df_rh['tempo_empresa_dias'] = (hoje - df_rh['data_contratacao']).dt.days

    # Selecionar colunas que importam
    df_rh_limpo = df_rh[[
        'id_funcionario', 
        'departamento', 
        'salario_mensal', 
        'tempo_empresa_dias', 
        'pediu_demissao'
    ]]
    return df_rh_limpo


def processar_dados_logs(path):
    """
    Carrega e *agrega* os dados de logs.
    - Calcula 'dias_desde_ultimo_login'
    - Calcula 'media_tempo_logado_min'
    """
    print(f"Processando Logs: {path}")
    df_logs = pd.read_csv(path)
    
    # 🧠 Explicação: Padronizar nomes de colunas
    # Os CSVs vieram com nomes diferentes ('user_id' vs 'id_funcionario')
    df_logs = df_logs.rename(columns={'user_id': 'id_funcionario'})
    df_logs['data_acesso'] = pd.to_datetime(df_logs['data_acesso'])
    hoje = pd.to_datetime(datetime.datetime.today().strftime("%Y-%m-%d"))

    # 🧠 Explicação: Agregação (Nível 2)
    # O arquivo de log tem MÚLTIPLAS linhas por funcionário.
    # Precisamos "achatar" (agregar) isso em UMA linha por funcionário.
    
    # 1. Feature: Dias desde o último login (um sinal de desengajamento)
    df_ultimo_login = df_logs.groupby('id_funcionario')['data_acesso'].max().reset_index()
    df_ultimo_login['dias_desde_ultimo_login'] = (hoje - df_ultimo_login['data_acesso']).dt.days

    # 2. Feature: Média de tempo logado (sinal de engajamento)
    df_media_tempo = df_logs.groupby('id_funcionario')['tempo_logado_min'].mean().reset_index()
    df_media_tempo = df_media_tempo.rename(columns={'tempo_logado_min': 'media_tempo_logado_min'})

    # Juntar as features de log
    df_logs_agg = pd.merge(
        df_ultimo_login[['id_funcionario', 'dias_desde_ultimo_login']], 
        df_media_tempo, 
        on='id_funcionario',
        how='outer' # Garante que não perdemos ninguém
    )
    return df_logs_agg


def processar_dados_suporte(path):
    """
    Carrega e *agrega* os dados de chamados de suporte.
    - Calcula 'total_chamados_suporte'
    """
    print(f"Processando Suporte: {path}")
    df_suporte = pd.read_csv(path)
    
    # Padronizar nome da coluna
    df_suporte = df_suporte.rename(columns={'id_func': 'id_funcionario'})

    # 🧠 Explicação: Agregação (Nível 2)
    # Contar quantos chamados (linhas) cada funcionário abriu.
    # Pessoas frustradas (prestes a sair) tendem a abrir mais chamados.
    df_suporte_agg = df_suporte.groupby('id_funcionario').size().reset_index(name='total_chamados_suporte')
    
    return df_suporte_agg


def main():
    """
    Função principal que orquestra toda a pipeline de ETL.
    """
    print("--- INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO (ETL) ---")
    
    # --- 1. EXTRACT & TRANSFORM ---
    df_rh = processar_dados_rh(PATH_RH)
    df_logs = processar_dados_logs(PATH_LOGS)
    df_suporte = processar_dados_suporte(PATH_SUPORTE)
    
    # --- 2. COMBINE (Join) ---
    print("Combinando as 3 fontes de dados...")
    
    # Começamos com a base de RH (df_rh), que é nossa "tabela fato"
    df_master = df_rh
    
    # Adicionamos os dados de logs
    # Usamos 'how='left'' para não perder funcionários que NUNCA logaram
    df_master = pd.merge(df_master, df_logs, on='id_funcionario', how='left')
    
    # Adicionamos os dados de suporte
    # Usamos 'how='left'' para não perder funcionários que NUNCA abriram chamado
    df_master = pd.merge(df_master, df_suporte, on='id_funcionario', how='left')

    # --- 3. Pós-Limpeza (Tratando Nulos do Join) ---
    # 🧠 Explicação: O 'left' join criou Nulos (NaN) para quem nunca logou
    # ou nunca abriu chamado. O modelo não aceita NaN.
    # Vamos preencher esses Nulos com valores lógicos:
    
    # Se nunca logou, os dias desde o último login é um nº alto (ex: 999 dias)
    df_master['dias_desde_ultimo_login'] = df_master['dias_desde_ultimo_login'].fillna(999)
    # Se nunca logou, a média de tempo é 0
    df_master['media_tempo_logado_min'] = df_master['media_tempo_logado_min'].fillna(0)
    # Se nunca abriu chamado, o total é 0
    df_master['total_chamados_suporte'] = df_master['total_chamados_suporte'].fillna(0)

    # --- 4. LOAD (Salvar o resultado) ---
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Garante que a pasta 'data' existe
    df_master.to_csv(PATH_OUTPUT, index=False)
    
    print("\n--- PIPELINE CONCLUÍDA COM SUCESSO! ---")
    print(f"DataFrame 'master' salvo em: {PATH_OUTPUT}")
    print("Amostra do DataFrame Master (limpo):")
    print(df_master.head())
    print("\nVerificando nulos (deve ser tudo 0):")
    print(df_master.info())

# 🧠 Explicação: Padrão de script Python (Lvl 2)
# O 'if __name__ == "__main__":' diz ao Python:
# "Só execute a função 'main()' se este arquivo for rodado diretamente"
# (Ex: 'python src/01_data_preprocessing.py')
# Se outro script 'importar' este, a função main() NÃO roda.
if __name__ == "__main__":
    main()