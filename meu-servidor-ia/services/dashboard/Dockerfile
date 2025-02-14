FROM python:3.10-slim

# Definir variáveis de ambiente padronizadas
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USER_NAME=appuser \
    APP_HOME=/app/dashboard

# Instalar dependências do sistema com otimização de camadas
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* && \
    groupadd -g 1000 ${USER_NAME} && \
    useradd -m -u 1000 -g ${USER_NAME} ${USER_NAME}

# Configurar diretório de trabalho
WORKDIR /app

# Copiar e instalar requirements com otimização de cache
COPY requirements.txt ${APP_HOME}/requirements.txt
RUN pip install --no-cache-dir -r ${APP_HOME}/requirements.txt

# Copiar código da aplicação com permissões corretas
COPY . ${APP_HOME}

# Criar diretórios com permissões seguras
RUN mkdir -p ${APP_HOME}/{logs,data,static,templates,cache} && \
    mkdir -p /app/shared && \
    chown -R ${USER_NAME}:${USER_NAME} /app && \
    chmod -R 750 ${APP_HOME} && \
    chmod -R 770 ${APP_HOME}/logs

# Mudar para o diretório da aplicação
WORKDIR ${APP_HOME}

# Mudar para usuário não-root
USER ${USER_NAME}

# Healthcheck otimizado
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Comando otimizado com configurações de produção
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "1000", "--timeout-keep-alive", "120"]