---
title: API de Previsão de Churn (Lvl 3)
sdk: docker
app_port: 8000 
---

# 🚀 Projeto de MLOps (Nível 3): API de Previsão de Churn

Este repositório contém o código de ponta-a-ponta para um projeto de People Analytics, que transforma dados brutos de RH em uma API de deploy em tempo real.

O modelo está (ou estará em breve) "vivo" nesta URL:
➡️ [https://huggingface.co/spaces/marcelofschiavo/api-churn-lvl3](https://huggingface.co/spaces/marcelofschiavo/api-churn-lvl3)

(A documentação da API estará em `/docs` no final da URL)

---

### 1. A Pipeline de Treinamento (Nível 2)

O modelo foi treinado com `scikit-learn`, `SMOTE` (para desbalanceamento) e `GridSearchCV` (para otimização de `recall`), e os experimentos foram registrados com `MLflow`.

### 2. O Deploy da API (Nível 3)

O pipeline `.pkl` salvo é servido por uma API **FastAPI** (`src/03_api.py`) e containerizado com um **Dockerfile**.