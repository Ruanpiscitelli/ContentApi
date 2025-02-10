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
COPY services/image_generator/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim

# Criar usuário não-root
RUN groupadd -r appuser && useradd -r -g appuser -s /sbin/nologin appuser

WORKDIR /app

# Copiar pacotes Python instalados do builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copiar código do serviço
COPY services/image_generator ./services/image_generator
COPY shared ./shared

# Criar diretórios necessários
RUN mkdir -p /app/models /app/cache/images /app/uploads/images /app/logs && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app/models /app/cache /app/uploads /app/logs

# Mudar para usuário não-root
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Iniciar o serviço
CMD ["python", "-m", "uvicorn", "meu-servidor-ia.services.image_generator.app:app", "--host", "0.0.0.0", "--port", "8000"]