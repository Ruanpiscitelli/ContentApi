#!/bin/bash

# Verifica se o arquivo .env existe
if [ ! -f .env ]; then
    echo "Arquivo .env não encontrado. Criando a partir do exemplo..."
    cp .env.example .env
fi

# Carrega as variáveis de ambiente
set -a
source .env
set +a

# Limpa containers e cache antigos
docker compose down
docker system prune -f

# Reconstrói os serviços
docker compose build --no-cache
docker compose up -d 