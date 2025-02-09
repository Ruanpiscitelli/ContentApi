# Usar imagem base com CUDA e Python
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Configurações de ambiente
ENV TZ=America/Sao_Paulo
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar
COPY meu-servidor-ia/services/video_generator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código do serviço
COPY meu-servidor-ia/services/video_generator ./meu-servidor-ia/services/video_generator
COPY meu-servidor-ia/shared ./meu-servidor-ia/shared

# Criar diretórios e configurar permissões
RUN mkdir -p /app/models /app/cache/videos /app/uploads/videos \
    && chmod -R 755 /app/models /app/cache /app/uploads

# Script de entrada
RUN echo '#!/bin/bash\ncd /app/meu-servidor-ia/services/video_generator\nuvicorn app:app --host 0.0.0.0 --port 8000' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]