# Build stage
FROM python:3.12-slim-bookworm AS builder

# Instalar dependências de sistema
RUN apt-get update && apt-get install -y \
    ca-certificates \
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

ENV TZ=America/Sao_Paulo
ENV GRADIO_SERVER_NAME="0.0.0.0"  # Necessário para acesso via Docker
ENV GRADIO_SERVER_PORT="9000"     # Porta explícita
ENV GRADIO_ROOT_PATH=""           # Path base da API

RUN apt-get update && apt-get install -y \
    ca-certificates \
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

RUN python3 download_models.py

EXPOSE 8000
CMD ["uvicorn", "services.voice_generator.app:app", "--host", "0.0.0.0", "--port", "8000"]
