#!/bin/bash

# Verifica o ambiente
if [ "$1" = "prod" ]; then
    echo "Iniciando em modo produção (com GPU)..."
    export USE_GPU=true
    docker compose up
else
    echo "Iniciando em modo desenvolvimento (sem GPU)..."
    export USE_GPU=false
    # Remove configurações específicas de GPU para desenvolvimento local
    docker compose up --remove-orphans
fi 