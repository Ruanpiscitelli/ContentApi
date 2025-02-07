FROM python:3.10-slim

# Define o diretório de trabalho
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos necessários
COPY requirements.txt .
COPY app.py .
COPY config.py .

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Configuração de variáveis de ambiente
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expõe a porta da API
EXPOSE 8000

# Inicia a aplicação
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
