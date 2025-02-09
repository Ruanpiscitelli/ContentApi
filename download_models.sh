#!/usr/bin/env bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Criar estrutura de diretórios
mkdir -p models/{texto,imagem,voz,video}

echo -e "${GREEN}Estrutura de diretórios criada${NC}"

# Função para baixar modelo
download_model() {
    local repo=$1
    local target=$2
    
    if [ -d "$target" ]; then
        echo -e "${GREEN}Modelo $repo já existe em $target${NC}"
        return 0
    fi
    
    echo "Baixando $repo para $target..."
    git lfs install
    git clone --depth 1 "https://huggingface.co/$repo" "$target"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Download concluído: $repo${NC}"
    else
        echo -e "${RED}Erro ao baixar: $repo${NC}"
        return 1
    fi
}

# Lista de modelos para download manual
declare -a models=(
    "texto/MiniCPM-o-2_6:openbmb/MiniCPM-o-2_6"
    "texto/DeepSeek-R1-Distill-Qwen-32B-abliterated:huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated"
    "imagem/animagine-xl-4.0:cagliostrolab/animagine-xl-4.0"
    "voz/fish-speech-1.5:fishaudio/fish-speech-1.5"
)

# Download dos modelos
for model in "${models[@]}"; do
    IFS=':' read -r path repo <<< "$model"
    download_model "$repo" "models/$path"
done

echo -e "${GREEN}Processo de download concluído${NC}" 