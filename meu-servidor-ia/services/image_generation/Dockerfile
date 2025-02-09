# Stage 1: Builder
FROM python:3.10-slim as builder

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY meu-servidor-ia/services/image_generation/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

# Instalar dependências do sistema necessárias em runtime
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    libcudart11.0 \
    libnvinfer8 \
    libnvonnxparsers8 \
    libnvparsers8 \
    libnvinfer-plugin8 \
    && rm -rf /var/lib/apt/lists/*

# Configurar variáveis de ambiente para CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# Copiar pacotes Python instalados do builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copiar código do serviço
COPY meu-servidor-ia/services/image_generation ./meu-servidor-ia/services/image_generation
COPY meu-servidor-ia/shared ./meu-servidor-ia/shared

# Criar diretório para modelos e uploads e configurar permissões em uma única camada
RUN mkdir -p /app/models /app/uploads/images /app/cache/images && \
    chmod -R 755 /app/models && \
    chmod -R 755 /app/uploads && \
    chmod -R 755 /app/cache

# Copiar e configurar script de entrada
COPY meu-servidor-ia/services/image_generation/run.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Configurar health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Expor porta do serviço
EXPOSE 8002

# Definir ponto de entrada
ENTRYPOINT ["/entrypoint.sh"]