#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Iniciando setup do serviço dashboard...${NC}"

# Verificar se estamos no diretório correto
if [ ! -d "meu-servidor-ia/services/dashboard" ]; then
    echo -e "${RED}Erro: Diretório meu-servidor-ia/services/dashboard não encontrado${NC}"
    echo "Por favor, execute este script do diretório raiz do projeto"
    exit 1
fi

# Verificar se o Docker está rodando
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Erro: Docker não está rodando${NC}"
    echo "Por favor, inicie o Docker Desktop e tente novamente"
    exit 1
fi

# Verificar se a porta 8006 está disponível
if lsof -i :8006 > /dev/null; then
    echo -e "${RED}Erro: Porta 8006 já está em uso${NC}"
    echo "Por favor, libere a porta 8006 e tente novamente"
    exit 1
fi

echo -e "${GREEN}Construindo imagem do dashboard...${NC}"

# Construir a imagem
docker build -t dashboard-service -f meu-servidor-ia/services/dashboard/Dockerfile meu-servidor-ia/services/dashboard

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build concluído com sucesso!${NC}"
    echo -e "\nPara executar o serviço:"
    echo "docker run -p 8006:8000 -v $(pwd)/meu-servidor-ia/shared:/app/shared dashboard-service"
else
    echo -e "${RED}Erro durante o build${NC}"
    exit 1
fi 