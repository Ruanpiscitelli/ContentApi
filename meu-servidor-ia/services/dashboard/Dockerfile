# Stage 1: Builder
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
COPY meu-servidor-ia/services/dashboard/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

# Instalar dependências do sistema necessárias em runtime
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    sqlite3 \
    zlib1g \
    liblzma5 \
    libzstd1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar pacotes Python instalados do builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copiar código do serviço
COPY meu-servidor-ia/services/dashboard ./meu-servidor-ia/services/dashboard
COPY meu-servidor-ia/shared ./meu-servidor-ia/shared

# Criar diretórios necessários
RUN mkdir -p /app/data /app/logs /app/uploads /app/cache /app/temp

# Configurar permissões
RUN chmod -R 755 /app/data /app/logs /app/uploads /app/cache /app/temp

# Copiar e configurar script de entrada
COPY meu-servidor-ia/services/dashboard/run.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Configurar health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expor porta do serviço
EXPOSE 8000

# Definir ponto de entrada
ENTRYPOINT ["/entrypoint.sh"] 