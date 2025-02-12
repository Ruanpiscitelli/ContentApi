# Build stage
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libcap2-bin \
    && rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /build

# Copiar e instalar requirements
COPY services/dashboard/requirements.txt .
RUN python -m pip install --upgrade pip==23.0.1
RUN pip install --no-cache-dir \
    pydantic==1.10.12 \
    fastapi==0.95.2 \
    typing-extensions==4.12.2 \
    -r requirements.txt

# Runtime stage
FROM python:3.10-slim

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV USER_NAME=appuser
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar runtime deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 ${USER_NAME} && \
    mkdir -p /app/{logs,data,uploads} && \
    chown -R ${USER_NAME}:${USER_NAME} /app

WORKDIR /app

# Copiar dependências do builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copiar código da aplicação
COPY --chown=appuser:appuser ./services/dashboard /app/services/dashboard
COPY --chown=appuser:appuser ./shared /app/shared

# Mudar para usuário root para instalação
USER root

WORKDIR /app/services/dashboard

# Permissão para usar porta 80 sem sudo
RUN apt-get update && apt-get install -y libcap2-bin && \
    setcap 'cap_net_bind_service=+ep' /usr/local/bin/python3.10 && \
    chmod u+s /usr/local/bin/python3.10

# Garanta a criação do diretório de logs
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

# Voltar para usuário appuser após instalação
USER appuser 