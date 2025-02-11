# Build stage
FROM python:3.10-slim as builder

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./services/video_editor/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim

# Definir variáveis de ambiente
ENV PYTHONPATH=/app/services/video_editor
ENV PYTHONUNBUFFERED=1
ENV USER_NAME=appuser

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root
RUN useradd -m -u 1000 ${USER_NAME}

# Configurar diretório de trabalho
WORKDIR /app

# Copiar arquivos do projeto
COPY --chown=appuser:appuser . /app/

# Criar diretórios e ajustar permissões
RUN mkdir -p /app/cache /app/models /app/storage /app/logs && \
    chown -R ${USER_NAME}:${USER_NAME} /app && \
    chmod -R 755 /app/cache /app/models /app/storage /app/logs

# Instalar dependências Python
RUN pip install --no-cache-dir -r services/video_editor/requirements.txt

# Mudar para o diretório do serviço
WORKDIR /app/services/video_editor

# Mudar para usuário não-root
USER ${USER_NAME}

# Healthcheck com tempo maior de start
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]