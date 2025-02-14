# Build stage
FROM python:3.10-slim as builder

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar o arquivo requirements.txt para o diretório atual (build context)
COPY services/video_editor/requirements.txt .

# Instalar as dependências do Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim

# Criar usuário não-root
RUN groupadd -r appuser && useradd -r -g appuser -s /sbin/nologin appuser && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar pacotes Python do builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copiar o restante do código do video_editor para a imagem
COPY --chown=appuser:appuser services/video_editor /app/services/video_editor
COPY --chown=appuser:appuser shared /app/shared

USER appuser

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

WORKDIR /app/services/video_editor

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]