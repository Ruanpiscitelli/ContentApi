#!/usr/bin/env bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Função para verificar se modelo já existe
check_model() {
    local path=$1
    if [ -d "$path" ]; then
        echo -e "${GREEN}Modelo já existe em $path${NC}"
        return 0
    fi
    return 1
}

# Função para baixar modelo
download_model() {
    local repo=$1
    local target=$2
    
    # Verifica se já existe
    if check_model "$target"; then
        return 0
    fi
    
    echo "Baixando $repo para $target..."
    
    # Instala git-lfs se necessário
    if ! command -v git-lfs &> /dev/null; then
        echo "Instalando git-lfs..."
        apt-get update && apt-get install -y git-lfs
        git lfs install
    fi
    
    # Baixa usando git-lfs
    git clone --depth 1 "https://huggingface.co/$repo" "$target"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Download concluído: $repo${NC}"
        # Baixa arquivos adicionais necessários
        if [ -f "$target/config.json" ]; then
            echo "Baixando arquivos adicionais..."
            cd "$target"
            git lfs pull
            cd -
        fi
        return 0
    else
        echo -e "${RED}Erro ao baixar: $repo${NC}"
        return 1
    fi
}

# Criar estrutura de diretórios
mkdir -p models/{texto,imagem,voz,video}

echo -e "${GREEN}Estrutura de diretórios criada${NC}"

# Lista de modelos para download
declare -a models=(
    "texto/MiniCPM-o-2_6:openbmb/MiniCPM-o-2_6"
    "texto/DeepSeek-R1-Distill-Qwen-32B-abliterated:huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated"
    "imagem/animagine-xl-4.0:cagliostrolab/animagine-xl-4.0"
    "voz/fish-speech-1.5:fishaudio/fish-speech-1.5"
    "video/stable-video-diffusion-img2vid-xt:stabilityai/stable-video-diffusion-img2vid-xt"
)

# Download dos modelos
for model in "${models[@]}"; do
    IFS=':' read -r path repo <<< "$model"
    download_model "$repo" "models/$path"
done

echo -e "${GREEN}Processo de download concluído${NC}" 