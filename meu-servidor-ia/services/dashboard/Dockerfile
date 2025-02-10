# Build stage
FROM python:3.10-slim as builder

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY services/dashboard/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim

# Instalar dependências runtime
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    sqlite3 \
    zlib1g \
    liblzma5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar pacotes Python instalados do builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copiar código do serviço
COPY services/dashboard ./services/dashboard
COPY shared ./shared

# Criar diretórios necessários
RUN mkdir -p /app/data /app/logs /app/uploads /app/cache /app/temp && \
    chmod -R 755 /app/data /app/logs /app/uploads /app/cache /app/temp

# Criar script de entrada
RUN echo '#!/bin/bash\n\
cd /app/services/dashboard\n\
if python3 -c "import uvloop" 2>/dev/null; then\n\
    exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools\n\
else\n\
    exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1\n\
fi' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Iniciar o serviço
ENTRYPOINT ["/entrypoint.sh"] 