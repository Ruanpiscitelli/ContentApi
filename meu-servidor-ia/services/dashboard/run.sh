#!/bin/bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Função para exibir mensagens
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Verificar se o Docker está instalado
if ! command -v docker &> /dev/null; then
    error "Docker não está instalado. Por favor, instale o Docker primeiro."
    exit 1
fi

# Verificar se o Redis está rodando
if ! docker ps | grep -q redis; then
    warn "Redis não está rodando. Iniciando container do Redis..."
    docker run -d --name redis -p 6379:6379 redis:latest
fi

# Construir a imagem
log "Construindo a imagem do dashboard..."
docker build -t dashboard-service .

# Verificar se a build foi bem sucedida
if [ $? -ne 0 ]; then
    error "Falha ao construir a imagem Docker"
    exit 1
fi

# Parar container existente se estiver rodando
if docker ps | grep -q dashboard-service; then
    log "Parando container existente..."
    docker stop dashboard-service
    docker rm dashboard-service
fi

# Criar rede Docker se não existir
if ! docker network ls | grep -q dashboard-network; then
    log "Criando rede Docker..."
    docker network create dashboard-network
fi

# Executar o container
log "Iniciando o servidor do dashboard..."
docker run -d \
    --name dashboard-service \
    -p 8000:8000 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    --network dashboard-network \
    dashboard-service

# Verificar se o container está rodando
if [ $? -eq 0 ]; then
    log "Servidor iniciado com sucesso!"
    log "Acesse o dashboard em: http://localhost:8000"
else
    error "Falha ao iniciar o container"
    exit 1
fi

# Mostrar logs
log "Mostrando logs do container..."
docker logs -f dashboard-service 