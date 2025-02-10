# Build stage
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Instalar Python e dependências de build
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    git-lfs \
    build-essential \
    ninja-build \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Instalar torch primeiro
RUN pip3 install --no-cache-dir torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY meu-servidor-ia/services/video_editor/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Runtime stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV TZ=America/Sao_Paulo
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    git-lfs \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY meu-servidor-ia/services/video_editor/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar código do serviço
COPY meu-servidor-ia/services/video_editor ./meu-servidor-ia/services/video_editor
COPY meu-servidor-ia/shared ./meu-servidor-ia/shared

# Criar diretórios necessários
RUN mkdir -p /app/models /app/cache/videos /app/uploads/videos /app/logs

# Configurar permissões
RUN chown -R nobody:nogroup /app && \
    chmod -R 755 /app/models /app/cache /app/uploads /app/logs

# Criar script de entrada
RUN echo '#!/bin/bash\n\
cd /app/meu-servidor-ia/services/video_editor\n\
if python3 -c "import uvloop" 2>/dev/null; then\n\
    exec python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools\n\
else\n\
    exec python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1\n\
fi' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Mudar para usuário não-root
USER nobody

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]