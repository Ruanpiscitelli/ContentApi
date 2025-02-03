FROM python:3.10-slim

# Atualiza os pacotes do sistema e instala dependências essenciais
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências
COPY requirements.txt .

# Atualiza o pip e instala as dependências Python
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia todo o código da aplicação para o diretório de trabalho
COPY . .

# Comando para iniciar o serviço com Gunicorn e UvicornWorker (4 workers)
CMD ["gunicorn", "app:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "4"]