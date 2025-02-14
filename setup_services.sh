#!/bin/bash

# Criar estrutura de diretórios base
mkdir -p services/{image_generator,voice_generator,video_generator,text_generation,video_editor,dashboard}
mkdir -p shared
mkdir -p models
mkdir -p storage
mkdir -p cache/{transformers,torch,huggingface}
mkdir -p logs/{image,voice,video,text,editor,dashboard}
mkdir -p uploads
mkdir -p temp

# Copiar arquivos do meu-servidor-ia para a nova estrutura
if [ -d "meu-servidor-ia" ]; then
    # Copiar serviços
    for service in image_generator voice_generator video_generator text_generation video_editor dashboard; do
        if [ -d "meu-servidor-ia/services/$service" ]; then
            cp -r meu-servidor-ia/services/$service/* services/$service/
        fi
    done

    # Copiar shared
    if [ -d "meu-servidor-ia/shared" ]; then
        cp -r meu-servidor-ia/shared/* shared/
    fi
fi

# Garantir permissões corretas
chmod -R 755 services
chmod -R 755 shared
chmod -R 755 models
chmod -R 755 storage
chmod -R 755 cache
chmod -R 755 logs
chmod -R 755 uploads
chmod -R 755 temp

echo "Estrutura de diretórios criada com sucesso!" 