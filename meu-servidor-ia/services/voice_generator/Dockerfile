# syntax=docker/dockerfile:1

# Stage de build
FROM python:3.10-slim AS builder

# Previne Python de gerar arquivos .pyc e mantém o stdout/stderr sem buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Instalar dependências de compilação
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Criar e ativar ambiente virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Copiar e instalar dependências usando cache mount
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel && \
    pip install torch==2.1.2+cpu torchaudio==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Baixar modelos usando cache mount
RUN --mount=type=cache,target=/root/.cache/huggingface \
    pip install --no-cache-dir huggingface_hub && \
    huggingface-cli download --resume-download fishaudio/fish-speech-1.5 --local-dir models/fish-speech-1.5

# Imagem final otimizada
FROM python:3.10-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Criar usuário não-root com UID específico para melhor segurança
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Instalar apenas runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    libsox-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copiar ambiente virtual e modelos do builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/models /app/models

WORKDIR /app

# Copiar código da aplicação e ajustar permissões
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "app:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "300"]