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
COPY meu-servidor-ia/services/voice_generator/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

# Instalar dependências do sistema necessárias em runtime
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libsndfile1 \
    libasound2 \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libpulse0 \
    libjack0 \
    libasound2-plugins \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar pacotes Python instalados do builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copiar código do serviço
COPY meu-servidor-ia/services/voice_generator ./meu-servidor-ia/services/voice_generator
COPY meu-servidor-ia/shared ./meu-servidor-ia/shared

# Criar diretório para modelos e áudios
RUN mkdir -p /app/models /app/uploads/audio /app/cache/audio

# Configurar permissões
RUN chmod -R 755 /app/models /app/uploads /app/cache

# Copiar e configurar script de entrada
COPY meu-servidor-ia/services/voice_generator/run.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Configurar health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Expor porta do serviço
EXPOSE 8003

# Definir ponto de entrada
ENTRYPOINT ["/entrypoint.sh"]