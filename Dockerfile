# Stage 1: Builder
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

# Evitar interações durante a instalação
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Instalar dependências essenciais e configurar repositório NVIDIA
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    curl \
    ninja-build \
    gnupg2 \
    ca-certificates \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin -o /etc/apt/preferences.d/cuda-repository-pin-600 \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring.deb \
    && dpkg -i cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-1 \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring.deb

WORKDIR /build

# Instalar PyTorch com CUDA
RUN pip3 install --no-cache-dir torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
    python3-pip \
    curl \
    ffmpeg \
    libsndfile1 \
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

# Copiar apenas os pacotes Python necessários
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copiar código
COPY . .

# Configurar diretórios com permissões corretas
RUN mkdir -p /app/models/cache /app/models/hub /app/data /app/logs && \
    chmod -R 777 /app/models /app/data /app/logs

# Configurações HuggingFace
ENV TRANSFORMERS_CACHE=/app/models/cache
ENV HF_HOME=/app/models/hub

# Otimizações CUDA
ENV CUDA_MODULE_LOADING=LAZY
ENV TORCH_ALLOW_TF32=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=all
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

# Healthcheck mais robusto para cloud
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Porta
EXPOSE 8000

# Inicialização otimizada para cloud
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "1000", "--timeout-keep-alive", "120"]