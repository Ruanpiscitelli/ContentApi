#!/bin/bash

# Criar estrutura de diretórios base
mkdir -p meu-servidor-ia/{shared,services}/{text_generation,image_generator,voice_generator,video_generator}/

# Criar diretórios para dados
mkdir -p {models,storage,logs/{video,image,text,voice},cache}

# Criar __init__.py nos diretórios principais
touch meu-servidor-ia/shared/__init__.py
for service in text_generation image_generator voice_generator video_generator; do
    touch meu-servidor-ia/services/${service}/__init__.py
done 