FROM python:3.10-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de requisitos
COPY requirements-base.txt requirements.txt ./

# Instala primeiro as dependências base (PyTorch e fundamentais)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-base.txt

# Instala o restante das dependências e adiciona gunicorn
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copia o código da aplicação
COPY . .

# Expõe a porta que o serviço usará
EXPOSE 8000

# Comando para iniciar o serviço com Gunicorn e workers Uvicorn
CMD ["gunicorn", "app:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300", "--keep-alive", "5", "--log-level", "info"]