# ---------------------------------------------------------------
# PROJETO 1 (NÍVEL 3) - ARQUIVO 4: Dockerfile
#
# OBJETIVO: A "receita" para construir uma "caixa" (Container)
# que contém nossa API, nosso modelo e todas as dependências.
# ---------------------------------------------------------------

# --- 1. A IMAGEM BASE (O Ponto de Partida) ---
# 🧠 Explicação: Não começamos do zero. Pegamos uma "imagem" oficial
# do Python (versão 3.10, "slim" é uma versão leve).
# Esta imagem já tem Python e 'pip' instalados.
FROM python:3.13-slim

# --- 2. O DIRETÓRIO DE TRABALHO (A "Oficina" na Caixa) ---
# 🧠 Explicação: Dentro da "caixa", vamos criar uma pasta /app
# onde todo o nosso código vai viver.
WORKDIR /app

# --- 3. A "LISTA DE COMPRAS" (Instalando Dependências) ---
# 🧠 Explicação: Copia SÓ a lista de compras primeiro. O Docker
# é inteligente (usa cache). Se a lista não mudar, ele não
# reinstala tudo, tornando a 'build' mais rápida.
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# --- 4. COPIANDO O PROJETO (Colocando o Código na Caixa) ---
# 🧠 Explicação: Agora sim, copia os "ingredientes" do nosso projeto
# para dentro da pasta /app na "caixa".
#
# Copia o 'best_churn_pipeline.pkl' para '/app/models/'.
COPY ./models/best_churn_pipeline.pkl ./models/
# Copia o '03_api.py' para '/app/src/'.
COPY ./src/03_api.py ./src/

# --- 5. EXPOR A PORTA (Abrindo a Janela da Loja) ---
# 🧠 Explicação: A API (uvicorn) vai rodar *dentro* da caixa na
# porta 8000. Precisamos "abrir um buraco" na parede da caixa
# para o mundo exterior poder falar com ela.
EXPOSE 8000

# --- 6. O COMANDO DE PARTIDA (A Chave de Ignição) ---
# 🧠 Explicação: O que a "caixa" deve fazer assim que for ligada?
# "Ligue o servidor uvicorn, escute em todas as IPs (0.0.0.0),
# na porta 8000, e sirva a API que está no arquivo 'src/03_api.py'
# e se chama 'app'".
CMD ["uvicorn", "src.03_api:app", "--host", "0.0.0.0", "--port", "7860"]