FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Configurar timezone
ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependências básicas primeiro
COPY requirements.txt .
RUN pip3 install --no-cache-dir packaging setuptools wheel && \
    pip3 install --no-cache-dir torch==2.2.1 && \
    pip3 install --no-cache-dir -r requirements.txt

# Copiar arquivos necessários
COPY download_models.py .
COPY services/voice_generator ./services/voice_generator
COPY shared ./shared

# Baixar modelos
RUN python3 download_models.py

# Expor porta
EXPOSE 8000

# Comando para iniciar o serviço
CMD ["uvicorn", "services.voice_generator.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Build stage
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel AS builder

# Instalar git para clonar repositórios
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY services/voice_generator/requirements.txt .

# Instalar com flags específicas para CUDA
RUN TORCH_CUDA_ARCH_LIST="8.6" pip install --no-cache-dir -r requirements.txt
