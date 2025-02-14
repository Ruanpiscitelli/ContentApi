#!/bin/bash
set -e

# Criar diretórios
mkdir -p cache/{videos,images}

# Ajustar permissões mais seguras
chown -R 1000:1000 cache
chmod -R 755 cache

# Verificar dependências
if ! command -v docker-compose &> /dev/null; then
    echo "docker-compose não encontrado"
    exit 1
fi

# Subir serviços
docker-compose up --build -d

# Aguardar serviços
echo "Aguardando serviços iniciarem..."
sleep 10

# Verificar status
docker-compose ps