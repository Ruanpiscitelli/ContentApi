# Build stage
FROM --platform=linux/amd64 pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar dependências de compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage final
FROM --platform=linux/amd64 pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Configurações de ambiente
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"  # Otimizado para RTX 4090

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    libsm6 \
    libxext6 \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar dependências e código
COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages
COPY . .

# Criar diretórios
RUN mkdir -p /app/models/cache /app/models/hub /app/data /app/logs && \
    chmod -R 777 /app/models /app/data /app/logs

# Configurações HuggingFace
ENV TRANSFORMERS_CACHE=/app/models/cache
ENV HF_HOME=/app/models/hub

# Otimizações CUDA
ENV CUDA_MODULE_LOADING=LAZY
ENV TORCH_ALLOW_TF32=1
ENV NCCL_P2P_DISABLE=1  # Melhora comunicação multi-GPU

# Portas
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint com 9 workers (2 * num_gpus + 1)
CMD ["gunicorn", "app:app", \
     "--workers", "9", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "300", \
     "--keep-alive", "120", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]