FROM python:3.10-slim

# Definir variáveis de ambiente
ENV PYTHONPATH=/app/services/dashboard
ENV PYTHONUNBUFFERED=1
ENV USER_NAME=appuser
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root
RUN useradd -m -u 1000 ${USER_NAME}

# Configurar diretório de trabalho
WORKDIR /app

# Copiar apenas os requirements primeiro
COPY --chown=appuser:appuser services/dashboard/requirements.txt /app/requirements.txt

# Instalar dependências Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar o resto dos arquivos do projeto
COPY --chown=appuser:appuser ./services/dashboard /app/services/dashboard
COPY --chown=appuser:appuser ./shared /app/shared

# Criar diretórios e ajustar permissões
RUN mkdir -p /app/services/dashboard/logs /app/services/dashboard/data /app/services/dashboard/static /app/services/dashboard/templates /app/services/dashboard/logs/api && \
    chown -R ${USER_NAME}:${USER_NAME} /app && \
    chmod -R 755 /app/services/dashboard/logs && \
    chmod -R 755 /app/services/dashboard/data && \
    chmod -R 755 /app/services/dashboard/static && \
    chmod -R 755 /app/services/dashboard/templates && \
    chmod -R 777 /app/services/dashboard/logs/api

# Mudar para o diretório do serviço
WORKDIR /app/services/dashboard

# Mudar para usuário não-root
USER ${USER_NAME}

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Inicializar o banco de dados e iniciar o servidor
CMD ["sh", "-c", "python -c 'from main import init_db; import asyncio; asyncio.run(init_db())' && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"] 