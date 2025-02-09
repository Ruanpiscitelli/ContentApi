# Build stage
FROM python:3.12-slim-bookworm AS builder

# Instalar dependências de sistema
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    git-lfs \
    libsox-dev \
    build-essential \
    cmake \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Instalar torch primeiro
RUN pip install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

COPY services/voice_generator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.12-slim-bookworm

# Configurações de ambiente
ENV TZ=America/Sao_Paulo
# Necessário para acesso via Docker
ENV GRADIO_SERVER_NAME="0.0.0.0"
# Porta explícita para Gradio
ENV GRADIO_SERVER_PORT="9000"
# Path base da API
ENV GRADIO_ROOT_PATH=""
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    git-lfs \
    libsox-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar dependências do build stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copiar arquivos do serviço
COPY download_models.py .
COPY services/voice_generator ./services/voice_generator
COPY shared ./shared

# Criar diretórios necessários
RUN mkdir -p /app/models/texto /app/models/voz /app/models/imagem /app/models/video

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

EXPOSE 9000

# Script de inicialização
COPY services/voice_generator/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
