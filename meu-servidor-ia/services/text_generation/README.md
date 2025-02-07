# Serviço de Geração de Texto

Este serviço fornece uma API compatível com OpenAI para geração de texto usando o vLLM.

## Estrutura do Serviço

```
services/text_generation/
├── app.py              # Aplicação FastAPI
├── config.py           # Configurações do serviço
├── requirements.txt    # Dependências Python
├── Dockerfile         # Configuração do container
├── k8s/               # Configurações Kubernetes
└── README.md          # Esta documentação
```

## Configuração

### Variáveis de Ambiente

- `API_TOKEN`: Token de autenticação da API
- `VLLM_ENDPOINT`: URL do serviço vLLM (default: http://vllm:8000/v1/completions)
- `TEXT_GEN_API_TOKEN`: Token específico para o serviço de geração de texto
- `REDIS_URL`: URL do Redis para cache
- `REDIS_PASSWORD`: Senha do Redis
- `API_PORT`: Porta da API (default: 8000)
- `METRICS_PORT`: Porta para métricas Prometheus (default: 8001)

### Modelos Suportados

O serviço suporta os seguintes modelos:
- mistralai/Mistral-7B-v0.1
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-chat-hf
- meta-llama/Llama-2-70b-chat-hf
- tiiuae/falcon-7b
- tiiuae/falcon-40b
- mosaicml/mpt-7b
- mosaicml/mpt-30b

## API Endpoints

### GET /v1/models

Lista todos os modelos disponíveis.

**Response:**
```json
{
    "object": "list",
    "data": [
        {
            "id": "mistralai/Mistral-7B-v0.1",
            "object": "model",
            "created": 1709913600,
            "owned_by": "organization"
        }
    ]
}
```

### GET /v1/models/{model_id}

Obtém informações sobre um modelo específico.

**Response:**
```json
{
    "model": "mistralai/Mistral-7B-v0.1",
    "status": "loaded",
    "gpu_memory_usage": 7.5,
    "loaded_at": 1709913600.0
}
```

### POST /v1/completions

Gera texto usando o modelo especificado.

**Request:**
```json
{
    "model": "mistralai/Mistral-7B-v0.1",
    "prompt": "Escreva um poema sobre",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.95,
    "n": 1,
    "stream": false,
    "stop": ["\n\n"],
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0
}
```

**Response:**
```json
{
    "id": "cmpl-abc123",
    "object": "text_completion",
    "created": 1709913600,
    "model": "mistralai/Mistral-7B-v0.1",
    "choices": [
        {
            "text": "o mar azul...",
            "index": 0,
            "logprobs": null,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 20,
        "total_tokens": 25
    }
}
```

### GET /health

Verifica o status do serviço.

**Response:**
```json
{
    "status": "ok",
    "service": "text-generator",
    "timestamp": 1709913600.0
}
```

## Cache Redis

O serviço utiliza Redis para cache das respostas com as seguintes configurações:
- TTL padrão: 1 hora
- Prefixo das chaves: "vllm:"
- Política de memória: allkeys-lru
- Limite de memória: 2GB

## Rate Limiting

O serviço implementa limites de requisições por modelo:
- Modelos 7B: 100 req/min
- Modelos 13B: 50 req/min
- Modelos 30B: 40 req/min
- Modelos 70B: 20 req/min

## Monitoramento

- Métricas Prometheus expostas na porta 8001
- Métricas disponíveis:
  - `text_generation_seconds`: Histograma do tempo de geração
  - `text_generation_errors_total`: Contador de erros
  - `text_cache_hits_total`: Contador de hits no cache
  - `text_cache_misses_total`: Contador de misses no cache
  - Métricas de GPU via py3nvml
  - Métricas de sistema via psutil
  - Métricas OpenTelemetry

## Docker

Build da imagem:
```bash
docker build -t text-generator .
```

Executar o container:
```bash
docker run -p 8006:8000 \
  -e API_TOKEN=seu-token \
  -e VLLM_ENDPOINT=http://vllm:8000/v1/completions \
  text-generator
```

## Kubernetes

Aplicar configurações:
```bash
kubectl apply -f k8s/text-generation-deployment.yaml
kubectl apply -f k8s/text-generation-service.yaml
```

## Integração vLLM

O serviço se integra com o vLLM através da interface OpenAI API. O vLLM deve estar rodando em um container separado, configurado através da variável de ambiente `VLLM_ENDPOINT`.

### Exemplo de uso do vLLM no vast.ai:

```bash
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai \
  --model mistralai/Mistral-7B-v0.1 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192
```

### Exemplo de chamada à API usando curl:

```bash
curl -X POST http://localhost:8006/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer seu-token" \
  -d '{
    "model": "mistralai/Mistral-7B-v0.1",
    "prompt": "Escreva um poema sobre o mar",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Exemplo de chamada à API usando Python:

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8006/v1/completions",
        headers={
            "Authorization": "Bearer seu-token"
        },
        json={
            "model": "mistralai/Mistral-7B-v0.1",
            "prompt": "Escreva um poema sobre o mar",
            "max_tokens": 100,
            "temperature": 0.7
        }
    )
    print(response.json())
``` 