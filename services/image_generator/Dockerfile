FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Configurar timezone
ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependências base primeiro
COPY services/image_generator/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY services/image_generator ./services/image_generator
COPY shared ./shared

# Criar diretório de modelos
RUN mkdir -p /app/models

# Expor porta
EXPOSE 8000

# Script de inicialização
COPY services/image_generator/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
