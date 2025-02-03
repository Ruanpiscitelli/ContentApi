FROM python:3.10-slim

# Atualiza o sistema e instala dependências essenciais
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    git \
 && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia o arquivo de dependências para o container
COPY requirements.txt .

# Atualiza o pip e instala as dependências Python listadas
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia todo o código da aplicação para o container
COPY . .

# Define o comando para iniciar o serviço utilizando Gunicorn com UvicornWorker (2 workers, pois o dashboard costuma ter carga menor)
CMD ["gunicorn", "app:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2"]