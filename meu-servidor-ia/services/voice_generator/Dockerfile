# Build stage
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

# Build arguments
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=America/Sao_Paulo

# Configurações de ambiente para build
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Instalar dependências de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libsndfile1-dev \
    libportaudio2 \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Instalar torch primeiro para otimizar cache
RUN pip3 install --no-cache-dir torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copiar e instalar requirements
COPY meu-servidor-ia/services/voice_generator/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Runtime stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Runtime arguments
ARG PYTHON_VERSION=3.10
ARG USER_NAME=appuser
ARG USER_UID=1000
ARG USER_GID=1000

# Configurações de ambiente para runtime
ENV TZ=America/Sao_Paulo \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    CUDA_VISIBLE_DEVICES=all \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Configurar timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instalar dependências runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Criar usuário não-root
RUN groupadd -g ${USER_GID} ${USER_NAME} && \
    useradd -u ${USER_UID} -g ${USER_GID} -s /bin/bash -m ${USER_NAME}

# Criar diretórios necessários
RUN mkdir -p /app/models /app/cache/voice /app/uploads/voice /app/logs && \
    chown -R ${USER_NAME}:${USER_NAME} /app

# Copiar pacotes Python do builder
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages /usr/local/lib/python${PYTHON_VERSION}/dist-packages

# Copiar código do serviço
COPY --chown=${USER_NAME}:${USER_NAME} meu-servidor-ia/services/voice_generator /app/meu-servidor-ia/services/voice_generator/
COPY --chown=${USER_NAME}:${USER_NAME} meu-servidor-ia/shared /app/meu-servidor-ia/shared/

# Criar script de entrada
RUN echo '#!/bin/bash\n\
cd /app/meu-servidor-ia/services/voice_generator\n\
\n\
# Verificar e criar diretórios necessários\n\
for dir in models cache/voice uploads/voice logs; do\n\
    if [ ! -d "/app/$dir" ]; then\n\
        echo "Criando diretório /app/$dir"\n\
        mkdir -p "/app/$dir"\n\
        chmod 755 "/app/$dir"\n\
    fi\n\
done\n\
\n\
# Aguardar dependências (se necessário)\n\
sleep 5\n\
\n\
# Iniciar o serviço\n\
if python3 -c "import uvloop" 2>/dev/null; then\n\
    exec python3 -m uvicorn app:app \\\n\
        --host 0.0.0.0 \\\n\
        --port 8000 \\\n\
        --workers 1 \\\n\
        --loop uvloop \\\n\
        --http httptools \\\n\
        --timeout-keep-alive 120 \\\n\
        --limit-concurrency 1000 \\\n\
        --log-level info\n\
else\n\
    exec python3 -m uvicorn app:app \\\n\
        --host 0.0.0.0 \\\n\
        --port 8000 \\\n\
        --workers 1 \\\n\
        --timeout-keep-alive 120 \\\n\
        --limit-concurrency 1000 \\\n\
        --log-level info\n\
fi' > /entrypoint.sh && \
    chmod +x /entrypoint.sh && \
    chown ${USER_NAME}:${USER_NAME} /entrypoint.sh

# Mudar para usuário não-root
USER ${USER_NAME}

# Healthcheck mais robusto
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]