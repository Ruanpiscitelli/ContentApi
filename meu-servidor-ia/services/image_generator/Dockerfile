FROM python:3.10-slim

# Atualiza os pacotes do sistema e instala dependências essenciais
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    git \
 && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia o arquivo de dependências para o container
COPY requirements.txt .

# Atualiza o pip e instala as dependências listadas
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia todo o código da aplicação para o diretório de trabalho
COPY . .

# Inicia o serviço com Gunicorn utilizando 4 workers (UvicornWorker para suporte assíncrono)
CMD ["gunicorn", "app:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "4"]