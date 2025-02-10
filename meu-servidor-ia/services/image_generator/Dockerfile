# Stage 1: Builder
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

# Evitar interações durante a instalação
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Instalar dependências essenciais
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar PyTorch com CUDA
RUN pip3 install --no-cache-dir torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copiar requirements e instalar dependências Python
COPY meu-servidor-ia/services/image_generator/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Configurações de ambiente
ENV TZ=America/Sao_Paulo
ENV PYTHONPATH=/app
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instalar runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    libgomp1 \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring.deb \
    && dpkg -i cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-cudart-12-1 \
        cuda-nvrtc-12-1 \
        cuda-nvtx-12-1 \
        libnvinfer8 \
        libnvonnxparsers8 \
        libnvparsers8 \
        libnvinfer-plugin8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring.deb

WORKDIR /app

# Criar usuário não-root
RUN groupadd -r appuser && useradd -r -g appuser -s /sbin/nologin appuser

# Copiar requirements e instalar dependências Python
COPY meu-servidor-ia/services/image_generator/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar código do serviço
COPY meu-servidor-ia/services/image_generator ./meu-servidor-ia/services/image_generator
COPY meu-servidor-ia/shared ./meu-servidor-ia/shared

# Criar diretórios necessários
RUN mkdir -p /app/models /app/cache/images /app/uploads/images /app/logs

# Configurar permissões
RUN chown -R appuser:appuser /app && \
    chmod -R 755 /app/models /app/cache /app/uploads /app/logs

# Criar e configurar script de entrada
RUN echo '#!/bin/bash\n\
cd /app/meu-servidor-ia/services/image_generator\n\
if python3 -c "import uvloop" 2>/dev/null; then\n\
    exec python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools\n\
else\n\
    exec python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1\n\
fi' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Mudar para usuário não-root
USER appuser

# Configurar health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expor porta do serviço
EXPOSE 8000

# Definir ponto de entrada
ENTRYPOINT ["/entrypoint.sh"]