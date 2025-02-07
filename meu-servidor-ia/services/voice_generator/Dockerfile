# syntax=docker/dockerfile:1.4
# ------------------------------------------------------------------
# Fish Speech Service - Otimizado para Mac
# https://speech.fish.audio/
# ------------------------------------------------------------------

# Stage 1: Base Python
FROM python:3.11-slim as base

# Configurações essenciais
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    ffmpeg \
    tini \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: Build do ambiente
FROM base AS builder

WORKDIR /app

# Criar e ativar ambiente virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar dependências em uma única camada
COPY requirements-base.txt requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-base.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copiar apenas os arquivos necessários
COPY app.py config.py fish_speech_wrapper.py ./
COPY shared ./shared/

# Stage Final: Runtime
FROM base AS runtime

WORKDIR /app

# Criar diretórios com permissões corretas
RUN mkdir -p /app/cache /app/temp /app/logs/tensorboard /tmp/numba_cache && \
    chmod -R 777 /app/cache /app/temp /app/logs /tmp/numba_cache

# Copiar apenas o necessário do builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Configurações de ambiente essenciais
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:${PYTHONPATH}" \
    NUMBA_CACHE_DIR="/tmp/numba_cache" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# Configurar usuário não-root para segurança
RUN useradd -r -u 1000 -g root appuser
USER appuser

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"] 