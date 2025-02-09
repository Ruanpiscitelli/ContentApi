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
    "imagem/ultimate-realistic-mix-v2-sdxl:John6666/ultimate-realistic-mix-v2-sdxl"
    "voz/fish-speech-1.5:fishaudio/fish-speech-1.5"
    "video/FastHunyuan:FastVideo/FastHunyuan-diffusers"
)

# Função para baixar do ModelScope
download_modelscope() {
    local model=$1
    local target=$2
    
    if check_model "$target"; then
        return 0
    fi
    
    echo "Baixando modelo do ModelScope: $model para $target..."
    
    # Instala modelscope_cli se necessário
    if ! command -v modelscope-cli &> /dev/null; then
        pip install modelscope_cli
    fi
    
    modelscope-cli download --model-name $model --save-dir $target
}

# Função para baixar do CivitAI
download_civitai() {
    local model_id=$1
    local target=$2
    
    if check_model "$target"; then
        return 0
    fi
    
    echo "Baixando modelo do CivitAI ID: $model_id para $target..."
    mkdir -p "$target"
    curl -L "https://civitai.com/api/download/models/$model_id" -o "$target/model.safetensors"
}

# Download dos modelos
for model in "${models[@]}"; do
    IFS=':' read -r path repo <<< "$model"
    case "$repo" in
        "FastVideo/FastHunyuan-diffusers")
            download_modelscope "$repo" "models/$path"
            ;;
        "John6666/ultimate-realistic-mix-v2-sdxl")
            download_civitai "266874" "models/$path"
            ;;
        *)
            download_model "$repo" "models/$path"
            ;;
    esac
done

echo -e "${GREEN}Processo de download concluído${NC}" 