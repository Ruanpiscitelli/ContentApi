FROM python:3.10-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copia os templates e arquivos estáticos primeiro
COPY templates/ templates/
COPY static/ static/

# Copia o restante dos arquivos
COPY . .

# Expõe a porta que o serviço irá usar
EXPOSE 8000

# Configurações do Gunicorn para produção
ENV WORKERS=4
ENV TIMEOUT=120
ENV KEEPALIVE=5

# Comando para iniciar o serviço com Gunicorn
CMD ["gunicorn", "app:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-"]