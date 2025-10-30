# ---------------------------------------------------------------
# PROJETO 1 (NÍVEL 3) - Dockerfile "Híbrido" (O Correto)
#
# Baseado na documentação do HFS, mas com a nossa versão do Python
# ---------------------------------------------------------------

# 1. A Base: Usando a *nossa* versão do Python (a do seu .venv)
FROM python:3.13-slim

# 2. Criando o usuário "user" (Boa prática do HFS)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

# 3. Instalando as Dependências (Como 'user')
COPY --chown=user requirements_api.txt .
RUN pip install --no-cache-dir --upgrade -r requirements_api.txt

# 4. Copiando nosso código (Como 'user')
COPY --chown=user ./models/best_churn_pipeline.pkl ./models/
COPY --chown=user app.py .

# 5. Expor a porta 7860 (Exigência do HFS)
EXPOSE 7860

# 6. O Comando de Partida (O "app:app" que o HFS espera)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]