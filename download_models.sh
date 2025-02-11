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
mkdir -p models/{text,image,voice,video}

echo -e "${GREEN}Estrutura de diretórios criada${NC}"

# Lista de modelos para download
declare -a models=(
    "text/MiniCPM-o-2_6:openbmb/MiniCPM-o-2_6"
    "text/DeepSeek-R1-Distill-Qwen-32B-abliterated:huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated"
    "image/animagine-xl-4.0:cagliostrolab/animagine-xl-4.0"
    "image/ultimate-realistic-mix-v2-sdxl:John6666/ultimate-realistic-mix-v2-sdxl"
    "voice/fish-speech-1.5:fish-speech-1.5"
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
    
    mkdir -p "$target"
    
    # Download direto dos arquivos necessários
    curl -L "https://modelscope.cn/api/v1/models/FastVideo/FastHunyuan-diffusers/repo?Revision=master&FilePath=model_index.json" \
         -o "$target/model_index.json"
    
    curl -L "https://modelscope.cn/api/v1/models/FastVideo/FastHunyuan-diffusers/repo?Revision=master&FilePath=pytorch_model.bin" \
         -o "$target/pytorch_model.bin"
    
    curl -L "https://modelscope.cn/api/v1/models/FastVideo/FastHunyuan-diffusers/repo?Revision=master&FilePath=config.json" \
         -o "$target/config.json"
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

# Função para baixar modelo Fish Speech
download_fish_speech() {
    local target=$1
    
    if check_model "$target"; then
        return 0
    fi
    
    echo "Baixando Fish Speech 1.5..."
    mkdir -p "$target"
    
    # Download dos três arquivos necessários
    curl -L "https://huggingface.co/fishaudio/fish-speech-1.5/resolve/main/model.pth?download=true" \
         -o "$target/model.pth"
    
    curl -L "https://huggingface.co/fishaudio/fish-speech-1.5/resolve/main/firefly-gan-vq-fsq-8x1024-21hz-generator.pth?download=true" \
         -o "$target/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    
    curl -L "https://huggingface.co/fishaudio/fish-speech-1.5/resolve/main/tokenizer.tiktoken?download=true" \
         -o "$target/tokenizer.tiktoken"
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
        "fish-speech-1.5")
            download_fish_speech "models/$path"
            ;;
        *)
            download_model "$repo" "models/$path"
            ;;
    esac
done

echo -e "${GREEN}Processo de download concluído${NC}"

# Text Generation
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2', cache_dir='models/text')"

# Image Generation
python3 -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', cache_dir='models/image')"

# Voice Generation
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/fastspeech2-en-ljspeech', cache_dir='models/voice')" 