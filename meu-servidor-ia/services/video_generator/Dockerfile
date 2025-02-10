# Usar imagem base com CUDA e Python
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Configurações de ambiente
ENV TZ=America/Sao_Paulo
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=all

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    build-essential \
    curl \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /app

# Criar estrutura de diretórios
RUN mkdir -p /app/models /app/cache/videos /app/uploads/videos /app/logs

# Copiar arquivos do projeto
COPY meu-servidor-ia/services/video_generator/requirements.txt /app/
COPY meu-servidor-ia/services/video_generator /app/meu-servidor-ia/services/video_generator/
COPY meu-servidor-ia/shared /app/meu-servidor-ia/shared/

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Configurar permissões
RUN chmod -R 755 /app/models /app/cache /app/uploads /app/logs && \
    chown -R nobody:nogroup /app/models /app/cache /app/uploads /app/logs

# Criar script de entrada
RUN echo '#!/bin/bash\n\
cd /app/meu-servidor-ia/services/video_generator\n\
if python -c "import uvloop" 2>/dev/null; then\n\
    exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools\n\
else\n\
    exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1\n\
fi' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Configurar usuário não-root
USER nobody

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expor porta
EXPOSE 8000

# Ponto de entrada
ENTRYPOINT ["/entrypoint.sh"]