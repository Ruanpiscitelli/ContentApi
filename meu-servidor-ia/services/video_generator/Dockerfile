# Stage 1: Builder
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS builder

# Evitar interações durante a instalação
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependências essenciais
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar e instalar requirements primeiro (para aproveitar cache)
COPY meu-servidor-ia/services/video_generator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Configurações de ambiente
ENV TZ=America/Sao_Paulo
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instalar runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar pacotes Python instalados do builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copiar código do serviço
COPY meu-servidor-ia/services/video_generator ./meu-servidor-ia/services/video_generator
COPY meu-servidor-ia/shared ./meu-servidor-ia/shared

# Criar diretórios necessários
RUN mkdir -p /app/models /app/cache/videos /app/uploads/videos \
    && chmod -R 755 /app/models /app/cache /app/uploads

# Criar e configurar script de entrada
RUN echo '#!/bin/bash\n\ncd /app/meu-servidor-ia/services/video_generator\nuvicorn app:app --host 0.0.0.0 --port 8000' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Configurar health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expor porta do serviço
EXPOSE 8000

# Definir ponto de entrada
ENTRYPOINT ["/entrypoint.sh"]