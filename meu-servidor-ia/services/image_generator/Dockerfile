# Build stage
FROM python:3.10-slim as builder

# Instalar dependências de sistema necessárias
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependências
COPY ./services/image_generator/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage final
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

# Copiar código do serviço
COPY --chown=appuser:appuser ./services/image_generator /app/services/image_generator
COPY --chown=appuser:appuser ./shared /app/shared

# Criar diretórios necessários
RUN mkdir -p /app/models /app/storage /app/logs /app/cache && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app/models /app/storage /app/logs /app/cache

USER appuser

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

WORKDIR /app/services/image_generator

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]