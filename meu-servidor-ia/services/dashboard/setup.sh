#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Iniciando setup do dashboard...${NC}"

# Verifica se python3 está instalado
if ! command -v python3 &> /dev/null; then
    echo "Python3 não encontrado. Por favor, instale o Python 3.8 ou superior."
    exit 1
fi

# Verifica se pip está instalado
if ! command -v pip3 &> /dev/null; then
    echo "pip3 não encontrado. Por favor, instale o pip."
    exit 1
fi

# Cria ambiente virtual se não existir
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Criando ambiente virtual...${NC}"
    python3 -m venv venv
fi

# Ativa o ambiente virtual
echo -e "${YELLOW}Ativando ambiente virtual...${NC}"
source venv/bin/activate

# Instala dependências
echo -e "${YELLOW}Instalando dependências...${NC}"
pip install -r requirements.txt

# Cria diretório de migrations se não existir
if [ ! -d "migrations" ]; then
    echo -e "${YELLOW}Inicializando Alembic...${NC}"
    alembic init migrations
fi

# Cria diretório para banco de dados se não existir
if [ ! -d "data" ]; then
    echo -e "${YELLOW}Criando diretório de dados...${NC}"
    mkdir data
fi

# Executa as migrations
echo -e "${YELLOW}Executando migrations...${NC}"
alembic upgrade head

echo -e "${GREEN}Setup concluído com sucesso!${NC}"
echo -e "${YELLOW}Para ativar o ambiente virtual use: source venv/bin/activate${NC}"
echo -e "${YELLOW}Credenciais padrão:${NC}"
echo -e "Usuário: admin"
echo -e "Senha: Admin@123" 