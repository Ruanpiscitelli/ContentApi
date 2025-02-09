FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Configurar timezone
ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar arquivos necessários
COPY requirements.txt .
COPY download_models.py .
COPY services/image_generator ./services/image_generator
COPY shared ./shared

# Instalar dependências Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Baixar modelos
RUN python3 download_models.py

# Expor porta
EXPOSE 8000

# Comando para iniciar o serviço
CMD ["uvicorn", "services.image_generator.app:app", "--host", "0.0.0.0", "--port", "8000"]
