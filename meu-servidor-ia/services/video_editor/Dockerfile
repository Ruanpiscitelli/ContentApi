FROM python:3.10-slim

# Atualiza os pacotes do sistema e instala dependências essenciais, incluindo FFmpeg para edição de vídeo
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    git \
 && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho no container
WORKDIR /app

# Copia o arquivo de dependências para o container
COPY requirements.txt .

# Atualiza o pip e instala as dependências listadas
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia todo o código do serviço para o diretório de trabalho
COPY . .

# Inicia o serviço utilizando Gunicorn com UvicornWorker; 2 workers geralmente são suficientes para um serviço de edição
CMD ["gunicorn", "app:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2"]
