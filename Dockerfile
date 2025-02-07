# Corrige a plataforma utilizada: se seu ambiente for ARM64, forçamos a utilização
FROM --platform=linux/arm64 pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# Explicação: Força o uso da plataforma ARM64, evitando conflito entre a arquitetura usada e a imagem base.

# Definir a variável de ambiente PYTHONPATH para evitar warning de variável indefinida
ENV PYTHONPATH=/app
# Explicação: Define PYTHONPATH para /app (ajuste conforme a estrutura do seu projeto).

# Evitar prompts durante a instalação
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências do sistema com --no-install-recommends para minimizar o tamanho
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    libsndfile1 \
    libportaudio2 \
    libsm6 \
    libxext6 \
    libjpeg-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /app

# Copiar arquivos de requisitos primeiro (boa prática para cache de layers)
COPY ./requirements.txt .

# Instalar dependências Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY . .

# Expor a porta da API
EXPOSE 8000

# Comando para iniciar a aplicação (corrigido para usar app.py)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Instalar apenas runtime dependencies com --no-install-recommends para minimizar dependências desnecessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libportaudio2 \
    libsm6 \
    libxext6 \
    libjpeg-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Explicação: A flag --no-install-recommends ajuda a evitar a instalação de pacotes extras desnecessários e possíveis conflitos. 