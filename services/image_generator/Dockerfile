# Build stage
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 as builder

# Configurar variáveis de ambiente
# TORCH_CUDA_ARCH_LIST="8.9" é específico para RTX 4090
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_CUDA_ARCHITECTURES="89"

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    cmake \
    git \
    ninja-build \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /build

# Instalar PyTorch com suporte CUDA 12.1
RUN pip3 install --no-cache-dir torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121

# Instalar otimizações para GPU
RUN pip3 install --no-cache-dir triton==2.1.0 xformers==0.0.23

# Copiar e instalar requirements
COPY ./requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Runtime stage
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Configurar variáveis de ambiente
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TRANSFORMERS_CACHE=/app/cache/transformers \
    TORCH_HOME=/app/cache/torch \
    HF_HOME=/app/cache/huggingface

# Instalar dependências de runtime
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

# Configurar diretórios da aplicação
WORKDIR /app
RUN mkdir -p /app/cache/{transformers,torch,huggingface} \
    && mkdir -p /app/{logs,uploads,temp,models} \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app/cache

# Copiar pacotes Python do builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Criar estrutura de pacotes Python
RUN mkdir -p /app/services/image_generator
COPY --chown=appuser:appuser ./ /app/services/image_generator/

# Mudar para usuário não-root
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expor porta
EXPOSE 8000

# Comando para executar a aplicação
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]