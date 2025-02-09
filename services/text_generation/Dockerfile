# Build stage
FROM python:3.10-slim-bookworm AS builder

# Instalar dependências de sistema
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    curl

WORKDIR /app

# Copiar requirements e instalar dependências
COPY services/text_generation/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar arquivos do serviço
COPY services/text_generation ./services/text_generation
COPY shared ./shared

# Criar diretório de modelos
RUN mkdir -p /app/models

# Healthcheck e configuração
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

EXPOSE 9000

# Script de inicialização
COPY services/text_generation/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
