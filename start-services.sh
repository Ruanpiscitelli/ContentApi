#!/bin/bash

# Função para verificar a saúde de um serviço
check_health() {
    local service=$1
    local max_attempts=30
    local attempt=1

    echo "Verificando saúde do serviço $service..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:$2/health" > /dev/null; then
            echo "Serviço $service está saudável!"
            return 0
        fi
        echo "Tentativa $attempt de $max_attempts para $service..."
        sleep 2
        ((attempt++))
    done
    echo "Falha ao verificar saúde do serviço $service"
    return 1
}

# Iniciar Redis primeiro
echo "Iniciando Redis..."
docker-compose up -d redis
sleep 5

# Iniciar serviços principais
echo "Iniciando geradores..."
docker-compose up -d image-generator voice-generator video-generator text-generation
sleep 10

# Verificar saúde dos serviços
check_health "image-generator" "8002"
check_health "voice-generator" "8003"
check_health "video-generator" "8004"
check_health "text-generation" "8005"

# Iniciar video-editor
echo "Iniciando editor de vídeo..."
docker-compose up -d video-editor
sleep 5
check_health "video-editor" "8005"

# Iniciar dashboard por último
echo "Iniciando dashboard..."
docker-compose up -d dashboard
sleep 5
check_health "dashboard" "8006"

echo "Todos os serviços foram iniciados!" 