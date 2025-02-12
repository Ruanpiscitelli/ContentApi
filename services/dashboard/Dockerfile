# Build stage
FROM python:3.10-slim as builder

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /build

# Copiar e instalar requirements
COPY services/dashboard/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir pydantic-settings==2.0.3 && \
    pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV USER_NAME=appuser
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar runtime deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 ${USER_NAME} && \
    mkdir -p /app/{logs,data,uploads} && \
    chown -R ${USER_NAME}:${USER_NAME} /app

WORKDIR /app

# Copiar dependências do builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copiar código da aplicação
COPY --chown=appuser:appuser ./services/dashboard /app/services/dashboard
COPY --chown=appuser:appuser ./shared /app/shared

# Mudar para usuário root para instalação
USER root

WORKDIR /app/services/dashboard

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

EXPOSE 80

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload", "--root-path", "/app/services/dashboard"]

# Voltar para usuário appuser após instalação
USER appuser 