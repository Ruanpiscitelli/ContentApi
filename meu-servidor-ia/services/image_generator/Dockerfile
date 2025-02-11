# Build stage
FROM python:3.10-slim as builder

# Instalar dependências de sistema necessárias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependências
COPY ./requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage final
FROM python:3.10-slim

# Criar usuário não-root
RUN groupadd -r appuser && useradd -r -g appuser -s /sbin/nologin appuser

WORKDIR /app

# Copiar pacotes Python do builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copiar código do serviço e shared
COPY . /app/services/image_generator/
COPY ../../shared /app/shared/

# Criar diretórios necessários
RUN mkdir -p /app/models /app/storage /app/logs && \
    chown -R appuser:appuser /app

# Mudar para usuário não-root
USER appuser

# Configurar variáveis de ambiente
ENV PYTHONPATH=/app

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Comando para iniciar o serviço
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]