#!/bin/bash

# Criar diretórios de cache se não existirem
mkdir -p cache/videos
mkdir -p cache/images

# Garantir permissões corretas
chmod -R 777 cache

# Subir todos os serviços
docker-compose up --build -d

# Mostrar status dos serviços
docker-compose ps 