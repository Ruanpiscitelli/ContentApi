#!/bin/bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Iniciando limpeza dos arquivos k8s...${NC}"

# Diretório k8s
K8S_DIR="../k8s"

# Verifica se o diretório k8s existe
if [ ! -d "$K8S_DIR" ]; then
    echo -e "${RED}Diretório k8s não encontrado!${NC}"
    exit 1
fi

# Cria backup dos arquivos k8s
echo -e "${YELLOW}Criando backup dos arquivos k8s...${NC}"
BACKUP_DIR="../k8s_backup_$(date +%Y%m%d_%H%M%S)"
cp -r "$K8S_DIR" "$BACKUP_DIR"

# Lista os arquivos que serão removidos
echo -e "${YELLOW}Os seguintes arquivos serão removidos:${NC}"
ls -l "$K8S_DIR"

# Pede confirmação
read -p "Deseja continuar com a remoção? (s/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Ss]$ ]]
then
    echo -e "${YELLOW}Operação cancelada.${NC}"
    exit 1
fi

# Remove os arquivos k8s
echo -e "${YELLOW}Removendo arquivos k8s...${NC}"
rm -rf "$K8S_DIR"

echo -e "${GREEN}Limpeza concluída!${NC}"
echo -e "${YELLOW}Backup dos arquivos k8s salvo em: $BACKUP_DIR${NC}"
echo -e "${GREEN}Os arquivos k0s permanecem intactos em: $(pwd)${NC}" 