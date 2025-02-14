#!/bin/bash

echo "Iniciando setup dos serviços..."

# Lista de serviços
SERVICES=("video_generator" "image_generator" "text_generator" "voice_generator" "video_editor" "dashboard")

# Criar estrutura de diretórios base
echo "Criando diretórios base..."
mkdir -p models
mkdir -p logs/{text,editor,dashboard}
mkdir -p cache/{transformers,huggingface}
mkdir -p uploads
mkdir -p temp
mkdir -p services

# Criar diretórios e arquivos para cada serviço
echo "Configurando serviços..."
for service in "${SERVICES[@]}"; do
    echo "Configurando $service..."
    
    # Criar diretório do serviço
    mkdir -p "services/$service"
    
    # Criar Dockerfile se não existir
    if [ ! -f "services/$service/Dockerfile" ]; then
        cat > "services/$service/Dockerfile" << EOL
FROM python:3.10-slim as builder

RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

RUN groupadd -r appuser && useradd -r -g appuser -s /sbin/nologin appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . /app/services/$service

USER appuser

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "services.$service.app:app", "--host", "0.0.0.0", "--port", "8000"]
EOL
    fi
    
    # Criar requirements.txt se não existir
    if [ ! -f "services/$service/requirements.txt" ]; then
        cp requirements.txt "services/$service/requirements.txt"
    fi
    
    # Criar app.py se não existir
    if [ ! -f "services/$service/app.py" ]; then
        cat > "services/$service/app.py" << EOL
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="$service Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "$service Service"}
EOL
    fi
    
    # Criar __init__.py
    touch "services/$service/__init__.py"
done

# Definir permissões
echo "Configurando permissões..."
chmod -R 755 models logs cache uploads temp services
find . -type f -name "*.sh" -exec chmod +x {} \;

# Criar arquivos .gitkeep
echo "Criando arquivos .gitkeep..."
find . -type d -empty -exec touch {}/.gitkeep \;

echo "Setup completo! Estrutura criada com sucesso."
echo "
Estrutura criada:
├── models/
├── logs/
│   ├── text/
│   ├── editor/
│   └── dashboard/
├── cache/
│   ├── transformers/
│   └── huggingface/
├── uploads/
├── temp/
└── services/
    ├── video_generator/
    ├── image_generator/
    ├── text_generator/
    ├── voice_generator/
    ├── video_editor/
    └── dashboard/
"