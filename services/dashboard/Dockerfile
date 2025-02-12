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
    && rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /build

# Copiar e instalar requirements
COPY services/dashboard/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Runtime stage
FROM python:3.10-slim

ENV PYTHONPATH=/app/services/dashboard
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

# Copiar wheels do builder
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copiar código da aplicação
COPY --chown=appuser:appuser ./services/dashboard /app/services/dashboard
COPY --chown=appuser:appuser ./shared /app/shared

# Mudar para usuário não-root
USER ${USER_NAME}

WORKDIR /app/services/dashboard

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 