FROM python:3.10-slim

# Definir variáveis de ambiente
ENV PYTHONPATH=/app
ENV USER_NAME=appuser

# Criar usuário não-root
RUN useradd -m -u 1000 ${USER_NAME} && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /app

# Copiar arquivos do projeto
COPY --chown=appuser:appuser . /app/

# Instalar dependências
RUN pip install --no-cache-dir -r services/video_editor/requirements.txt

# Criar diretórios específicos
RUN mkdir -p /app/cache/videos && \
    chown -R ${USER_NAME}:${USER_NAME} /app/cache

# Mudar para o diretório do serviço
WORKDIR /app/services/video_editor

# Mudar para usuário não-root
USER ${USER_NAME}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]