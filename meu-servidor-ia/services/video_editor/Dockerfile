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

COPY meu-servidor-ia/requirements/video_editor.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

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
    python3-pip \
    git \
    git-lfs \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar dependências do build stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copiar arquivos do serviço
COPY meu-servidor-ia/services/video_editor ./meu-servidor-ia/services/video_editor
COPY meu-servidor-ia/shared ./meu-servidor-ia/shared

# Criar diretório de modelos
RUN mkdir -p /app/models

# Script de inicialização
COPY meu-servidor-ia/services/video_editor/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"] 