# Stage 1: Base com CUDA
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 as base
ENV TARGETARCH=amd64

# Configurações essenciais
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}" \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    # Otimizações para RTX 4090
    TORCH_CUDA_ARCH_LIST="8.9" \
    CUDA_VISIBLE_DEVICES=all \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configurar Python e Pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Configurar Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

# Instalar dependências Python
COPY requirements.txt .
COPY requirements-cuda.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-cuda.txt

# Copia o código da aplicação
COPY . .

# Expõe a porta que o serviço usará
EXPOSE 8000

# Comando para iniciar o serviço com Gunicorn e workers Uvicorn
CMD ["gunicorn", "app:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300", "--keep-alive", "5", "--log-level", "info"]