#!/bin/bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Instalando kompose...${NC}"
# Instala kompose se não estiver instalado
if ! command -v kompose &> /dev/null; then
    curl -L https://github.com/kubernetes/kompose/releases/download/v1.31.2/kompose-darwin-amd64 -o kompose
    chmod +x kompose
    sudo mv kompose /usr/local/bin/
fi

echo -e "${YELLOW}Convertendo docker-compose.yml para k0s...${NC}"
cd ..
kompose convert -f docker-compose.yml -o k0s/services/

echo -e "${YELLOW}Ajustando manifestos gerados...${NC}"
cd k0s/services/

# Adiciona namespace aos manifestos
for file in *.yaml; do
    if [ -f "$file" ]; then
        echo "Processando $file..."
        # Adiciona namespace content-api
        sed -i '' 's/namespace: default/namespace: content-api/' "$file"
        # Adiciona labels padrão
        sed -i '' '/metadata:/a\  labels:\n    app: content-api' "$file"
    fi
done

echo -e "${GREEN}Conversão concluída! Arquivos gerados em k0s/services/${NC}" 