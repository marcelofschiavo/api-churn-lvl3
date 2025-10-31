# ---------------------------------------------------------------
# PROJETO 1 (NÍVEL 3) - Dockerfile FINAL (Versão da Documentação)
#
# Segue as regras de Permissão (USER user) e Python (3.9) do HFS.
# ---------------------------------------------------------------

# 1. A Base: Usando a versão de Python RECOMENDADA pelo HFS
FROM python:3.13-slim

# 2. Criando o usuário "user" (ID 1000) como a doc pede
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 3. O WORKDIR CORRETO: Dentro da casa do "user"
WORKDIR /home/user/app

# 4. Instalando as Dependências (Como 'user')
COPY --chown=user requirements_api.txt .
RUN pip install --no-cache-dir --upgrade -r requirements_api.txt

# 5. Copiando nosso código (Como 'user', para o WORKDIR correto)
COPY --chown=user ./models/best_churn_pipeline.pkl ./models/
COPY --chown=user app.py .

# 6. Expor a porta 7860 (Exigência do HFS)
EXPOSE 7860

# 7. O Comando de Partida (O "app:app" que o HFS espera)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]