FROM python:3.10-slim

# Configuração de variáveis de ambiente para otimização
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX" \
    FORCE_CUDA="1"

# Atualiza os pacotes do sistema e instala dependências essenciais
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    libasound2-dev \
    build-essential \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia o arquivo de dependências para o container
COPY requirements.txt .

# Atualiza o pip e instala as dependências Python
RUN pip install --upgrade pip && \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Baixa os modelos do Fish Speech
RUN mkdir -p models/fishespeech && \
    curl -L https://huggingface.co/fishaudio/fish-speech/resolve/main/config.json -o models/fishespeech/config.json && \
    curl -L https://huggingface.co/fishaudio/fish-speech/resolve/main/pytorch_model.bin -o models/fishespeech/pytorch_model.bin

# Copia todo o código da aplicação
COPY . .

# Expõe as portas do serviço e métricas
EXPOSE 8000 8001

# Define o comando para iniciar o serviço
CMD ["gunicorn", "app:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "300"]