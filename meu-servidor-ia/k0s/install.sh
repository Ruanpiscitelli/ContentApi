#!/bin/bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Iniciando instalação do k0s...${NC}"

# Verifica se o k0s já está instalado
if ! command -v k0s &> /dev/null; then
    echo -e "${YELLOW}Instalando k0s...${NC}"
    curl -sSLf https://get.k0s.sh | sudo sh
else
    echo -e "${GREEN}k0s já está instalado${NC}"
fi

# Verifica se o cluster já está rodando
if ! k0s status &> /dev/null; then
    echo -e "${YELLOW}Iniciando cluster k0s...${NC}"
    sudo k0s install controller --config k0s.yaml
    sudo k0s start
else
    echo -e "${GREEN}Cluster k0s já está rodando${NC}"
fi

# Aguarda o cluster iniciar
echo -e "${YELLOW}Aguardando cluster iniciar...${NC}"
sleep 30

# Configura o kubeconfig
echo -e "${YELLOW}Configurando kubeconfig...${NC}"
sudo k0s kubeconfig admin > ~/.kube/config
chmod 600 ~/.kube/config

# Cria namespace
echo -e "${YELLOW}Criando namespace content-api...${NC}"
kubectl create namespace content-api

# Configura secrets
echo -e "${YELLOW}Configurando secrets...${NC}"
# Converte variáveis de ambiente para base64
export TEXT_GEN_API_KEY_BASE64=$(echo -n "$TEXT_GEN_API_KEY" | base64)
export IMAGE_GEN_API_KEY_BASE64=$(echo -n "$IMAGE_GEN_API_KEY" | base64)
export VOICE_GEN_API_KEY_BASE64=$(echo -n "$VOICE_GEN_API_KEY" | base64)
export VIDEO_GEN_API_KEY_BASE64=$(echo -n "$VIDEO_GEN_API_KEY" | base64)
export VIDEO_EDIT_API_KEY_BASE64=$(echo -n "$VIDEO_EDIT_API_KEY" | base64)
export REDIS_PASSWORD_BASE64=$(echo -n "$REDIS_PASSWORD" | base64)
export MINIO_ACCESS_KEY_BASE64=$(echo -n "$MINIO_ACCESS_KEY" | base64)
export MINIO_SECRET_KEY_BASE64=$(echo -n "$MINIO_SECRET_KEY" | base64)
export DASHBOARD_USERNAME_BASE64=$(echo -n "$DASHBOARD_USERNAME" | base64)
export DASHBOARD_PASSWORD_BASE64=$(echo -n "$DASHBOARD_PASSWORD" | base64)

# Aplica as configurações
echo -e "${YELLOW}Aplicando configurações...${NC}"

# Aplica secrets
envsubst < secrets.yaml | kubectl apply -f -

# Aplica serviços de infraestrutura
echo -e "${YELLOW}Configurando serviços de infraestrutura...${NC}"
kubectl apply -f services/redis.yaml
kubectl apply -f services/minio.yaml

# Aguarda serviços de infraestrutura iniciarem
echo -e "${YELLOW}Aguardando serviços de infraestrutura...${NC}"
kubectl wait --for=condition=ready pod -l app=redis -n content-api --timeout=300s
kubectl wait --for=condition=ready pod -l app=minio -n content-api --timeout=300s

# Aplica serviços de IA
echo -e "${YELLOW}Configurando serviços de IA...${NC}"
kubectl apply -f services/text-generation.yaml
kubectl apply -f services/image-generator.yaml
kubectl apply -f services/voice-generator.yaml
kubectl apply -f services/video-generator.yaml
kubectl apply -f services/video-editor.yaml

# Aplica dashboard e monitoramento
echo -e "${YELLOW}Configurando dashboard e monitoramento...${NC}"
kubectl apply -f services/dashboard.yaml

# Aplica ingress
echo -e "${YELLOW}Configurando ingress...${NC}"
kubectl apply -f services/ingress.yaml

# Verifica status dos pods
echo -e "${YELLOW}Verificando status dos pods...${NC}"
kubectl get pods -n content-api

echo -e "${GREEN}Instalação concluída!${NC}"

# Instruções de acesso
echo -e "\n${GREEN}Instruções de Acesso:${NC}"
echo -e "1. Dashboard: https://api.ruanpiscitelli.com/dashboard"
echo -e "2. MinIO Console: https://minio.ruanpiscitelli.com"
echo -e "3. Métricas: https://api.ruanpiscitelli.com/metrics"

# Verifica serviços
echo -e "\n${YELLOW}Status dos Serviços:${NC}"
kubectl get services -n content-api

# Verifica ingress
echo -e "\n${YELLOW}Status do Ingress:${NC}"
kubectl get ingress -n content-api 