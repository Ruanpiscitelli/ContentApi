# Meu Servidor de IA para Geração de Conteúdo

Este projeto tem como objetivo construir um servidor cloud robusto que oferece diversas APIs para geração de conteúdo com inteligência artificial. Utilizando FastAPI, o sistema disponibiliza endpoints para:

- **Geração de Imagens** (usando modelos como SDXL/Flux com injeção de LoRAs);
- **Geração/Clonagem de Voz** (processamento de texto para áudio, com suporte à clonagem de voz via templates);
- **Geração de Vídeo com Hunyuan** (usando templates para definir parâmetros de geração);
- **Edição de Vídeo via JSON** (inspirado no [json2video](https://json2video.com/docs/tutorial/));
- **Dashboard Administrativo** para gerenciamento, upload de modelos, configuração de endpoints, visualização de logs e métricas.

O sistema foi pensado para ser altamente escalável – com execução assíncrona, uso máximo (e otimizado) de GPUs (inclusive multi‑GPU via DataParallel) – e containerizado, podendo ser orquestrado via Docker Compose durante o desenvolvimento e migrado para Kubernetes em produção (com Horizontal Pod Autoscaler e Cluster Autoscaler).

---

## Índice

- [Recursos](#recursos)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação e Configuração](#instalação-e-configuração)
  - [Desenvolvimento Local com Docker Compose](#desenvolvimento-local-com-docker-compose)
  - [Implantação no Kubernetes](#implantação-no-kubernetes)
- [Uso dos Endpoints](#uso-dos-endpoints)
- [Templates](#templates)
- [Contribuição](#contribuição)
- [Licença](#licença)

---

## Recursos

- **APIs Baseadas em FastAPI**: Cada serviço possui seu próprio endpoint e é escrito utilizando FastAPI, garantindo alta performance com suporte assíncrono.
- **Suporte a Templates**: Configurações pré-estabelecidas para cada fluxo (imagens, voz, vídeo) são definidas por arquivos JSON armazenados no diretório `templates/`.
- **Utilização de GPU**: O pipeline de geração de imagens é otimizado para uso em GPUs e suporta a execução paralela em múltiplas GPUs via `torch.nn.DataParallel`.
- **Escalabilidade**: Serviços containerizados que podem ser escalados via Kubernetes (com HPA e Cluster Autoscaler) para suportar alta concorrência.
- **Dashboard Administrativo**: Interface web para gerenciamento dos serviços, upload de modelos, configuração dos endpoints e monitoramento de logs e métricas.

## Atualizações Recentes

### Novas Funcionalidades
- Suporte a múltiplos schedulers na geração de imagens (DDIM, DPMSolver++, Euler, etc.)
- Sistema de cache em dois níveis (Redis + VRAM) para otimização de performance
- Integração com ControlNet para geração de imagens condicionadas
- Suporte a clonagem de voz com Fish Speech v2
- Templates pré-configurados para geração de shorts e vídeos verticais

### Otimizações de Performance
- Implementação de Torch Compile para JIT optimization
- Suporte a VAE Tiling para processamento de imagens em alta resolução
- Attention Slicing para otimização de uso de VRAM
- Gerenciamento dinâmico de recursos GPU com failover automático

### Mudanças na Estrutura
- Novo diretório `models/cache` para armazenamento local de modelos
- Templates reorganizados por categoria (imagem/voz/vídeo)
- Implementação de rate limiting baseado em GPU-hora
- Sistema de monitoramento com Prometheus/Grafana


---

## Tecnologias Utilizadas

- **Linguagem e Framework**: Python, FastAPI, Uvicorn, Gunicorn.
- **Modelos e Bibliotecas de IA**: Diffusers, Torch, Transformers.
- **Containerização**: Docker e Docker Compose.
- **Orquestração**: Kubernetes (para produção).
- **Proxy Reverso**: Nginx.
- **Monitoramento (Opcional)**: Prometheus e Grafana.
- **Templates de Configuração**: JSON (armazenados no diretório `templates/`).

---

## Estrutura do Projeto

```plaintext
meu-servidor-ia/
├── docker-compose.yml
├── nginx/
│   └── default.conf
├── k8s/                           # Arquivos YAML para implantação no Kubernetes (Deployments, Services, HPA, Ingress, etc.)
│   ├── dashboard-deployment.yaml
│   ├── dashboard-service.yaml
│   ├── image-generator-deployment.yaml
│   ├── image-generator-hpa.yaml
│   ├── image-generator-service.yaml
│   ├── voice-generator-deployment.yaml
│   ├── voice-generator-hpa.yaml
│   ├── voice-generator-service.yaml
│   ├── video-generator-deployment.yaml   # Geração de vídeo utilizando Hunyuan
│   ├── video-generator-hpa.yaml
│   ├── video-generator-service.yaml
│   ├── video-editor-deployment.yaml      # Edição de vídeo a partir de JSON (inspirado no json2video)
│   ├── video-editor-hpa.yaml
│   ├── video-editor-service.yaml
│   └── ingress.yaml
├── dashboard/                     # Serviço de Dashboard (GUI) para gerenciamento e upload/edição de modelos, visualização de logs e métricas
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── static/                    # Arquivos estáticos (CSS, JS, imagens, etc.)
│   └── templates/
│       └── dashboard.html
├── services/                      # Microserviços API utilizando FastAPI
│   ├── image_generator/           # API para geração de imagens (SDXL/Flux + LoRAs)
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── voice_generator/           # API para geração/clonagem de voz (Fishspeech e voice cloning)
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── video_generator/           # API para geração de vídeo utilizando Hunyuan
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── video_editor/              # API para edição de vídeo a partir de JSON (inspirado no json2video)
│       ├── app.py
│       ├── Dockerfile
│       └── requirements.txt
└── templates/                     # Templates JSON atualizados por categoria
    ├── image/
    │   ├── template1.json
    │   └── ultrarealistic.json
    ├── voice/
    │   ├── template1.json
    │   └── youtube.json
    └── video/
        ├── generator/
        │   └── template1.json
        └── editor/
            ├── template1.json
            └── shorts.json
```

## Exemplos de Uso Atualizados

### Geração de Imagem com ControlNet
```bash
curl -X POST http://localhost/generate-image \
  -H "Authorization: Bearer YOUR_SECURE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "cidade futurista",
    "template_id": "image/ultrarealistic",
    "controlnet": {
      "model_id": "lllyasviel/control_v11p_sd15_canny",
      "image_url": "https://example.com/reference.jpg",
      "preprocessor": "canny"
    }
  }'
```

### Clonagem de Voz com Fish Speech v2
```bash
curl -X POST http://localhost/generate-voice \
  -H "Authorization: Bearer YOUR_SECURE_TOKEN" \
  -F "texto=Olá, este é um teste de clonagem de voz" \
  -F "use_voice_clone=true" \
  -F "sample=@/path/to/reference.wav" \
  -F "template_id=voice/youtube"
```

### Geração de Shorts
```bash
curl -X POST http://localhost/edit-video \
  -H "Authorization: Bearer YOUR_SECURE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "video/editor/shorts",
    "input": "input_video.mp4",
    "format": "vertical",
    "resolution": "1080x1920"
  }'
```

---

## Instalação e Configuração

Desenvolvimento Local com Docker Compose
	1.	Clone o repositório:

git clone https://github.com/seu-usuario/meu-servidor-ia.git
cd meu-servidor-ia


	2.	Construa e inicie os containers:

docker-compose build
docker-compose up


	3.	Acesse os endpoints:
	•	As APIs estarão disponíveis nos respectivos ports definidos no docker-compose.yml.
	•	O Nginx irá encaminhar as requisições (ex.: http://localhost/generate-image).

Implantação no Kubernetes
	1.	Configure seu cluster Kubernetes (por exemplo, via vast.ai ou outro provedor) e certifique-se de que o NVIDIA Device Plugin e Metrics Server estão instalados.
	2.	Aplique os arquivos de deployment:

kubectl apply -f k8s/image-generator-deployment.yaml
kubectl apply -f k8s/image-generator-service.yaml
kubectl apply -f k8s/image-generator-hpa.yaml
# Repita para voice_generator, video_generator, video_editor e dashboard.
kubectl apply -f k8s/ingress.yaml   # (Opcional, para expor via Ingress)


	3.	Verifique os recursos:

kubectl get pods
kubectl get hpa
kubectl get nodes

---

## Uso dos Endpoints

Geração de Imagens
	•	Endpoint: /generate-image
	•	Requisição Exemplo (usando template):

curl -X POST http://localhost/generate-image \
  -H "Authorization: Bearer YOUR_SECURE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Uma paisagem futurista",
    "template_id": "image_template1"
  }'



Geração de Voz
	•	Endpoint: /generate-voice
	•	Requisição Exemplo:

curl -X POST http://localhost/generate-voice \
  -H "Authorization: Bearer YOUR_SECURE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "texto": "Olá, este é um teste de clonagem de voz.",
    "template_id": "voice_template1"
  }'



Geração de Vídeo (Hunyuan)
	•	Endpoint: /generate-video
	•	Requisição Exemplo:

curl -X POST http://localhost/generate-video \
  -H "Authorization: Bearer YOUR_SECURE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "video_generator_template1",
    "prompt": "Uma cidade futurista durante o pôr do sol",
    "override": {"duration": 15}
  }'



Edição de Vídeo (JSON)
	•	Endpoint: /edit-video
	•	Requisição Exemplo:

curl -X POST http://localhost/edit-video \
  -H "Authorization: Bearer YOUR_SECURE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "video_editor_template1",
    "override": {
       "input": "input_video.mp4",
       "operations": [
           {"action": "trim", "start": 5, "end": 20},
           {"action": "overlay", "file": "logo.png", "position": "top-right"}
       ],
       "output": {"file": "edited_video.mp4", "codec": "libx264"}
    }
}'

---

## Templates

Os templates são arquivos JSON armazenados no diretório templates/ e definem os parâmetros dos pipelines.
Exemplo de image_template1.json:

{
  "template_id": "image_template1",
  "model": "sdxl",
  "num_inference_steps": 60,
  "loras": [
    {"path": "loras/lora_face.pt", "scale": 0.8},
    {"path": "loras/lora_style.pt", "scale": 1.0}
  ]
}

Os templates para voz, vídeo de geração e edição seguirão estruturas semelhantes, contendo os parâmetros necessários para cada fluxo.

---

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.
Por favor, mantenha as convenções e a estrutura do projeto ao adicionar novas funcionalidades.

---

## Licença

Este projeto está licenciado sob a MIT License.

---

## Contato

Para dúvidas ou sugestões, entre em contato através do seu-email@dominio.com.

## Variáveis de Ambiente Adicionais
```yaml
TORCH_COMPILE_MODE: "reduce-overhead"  # Otimização JIT
ENABLE_VAE_TILING: "true"             # Para imagens em alta resolução
GPU_RATE_LIMIT: "100"                 # Requisições por GPU-hora
```

This README.md provides a complete description of the project, technology, structure, installation, usage, and further details.