# Serviço de Geração de Voz

Serviço de geração de voz baseado no Fish Speech com suporte a múltiplos backends e fallback automático.

## Funcionalidades

- Síntese de voz em múltiplos idiomas
- Clonagem de voz a partir de áudio de referência
- Processamento em batch para melhor performance
- Sistema de cache para resultados e embeddings
- Fallback automático entre backends
- Monitoramento detalhado de performance
- **Novo**: Suporte a WebSocket para streaming de áudio
- **Novo**: Validação avançada de parâmetros com Pydantic
- **Novo**: Tratamento robusto de erros
- **Novo**: Utilitários otimizados para I/O de áudio

## Backends Suportados

### Fish Audio API
- Serviço em nuvem para geração de voz
- Requer chave de API
- Ideal para produção e alta demanda
- **Novo**: Suporte a streaming via WebSocket

### Fish Speech Local
- Modelo local para geração de voz
- Suporte a GPU com otimizações FP16
- Ideal para desenvolvimento e baixa latência
- **Novo**: Cache de embeddings para clonagem

## Configuração

### Variáveis de Ambiente

```bash
# API e autenticação
API_TOKEN=seu_token_aqui
FISH_AUDIO_API_KEY=sua_chave_aqui

# Configurações do servidor
API_HOST=0.0.0.0
API_PORT=8000

# Redis (opcional)
REDIS_URL=redis://redis:6379

# MinIO (opcional)
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=seu_access_key
MINIO_SECRET_KEY=seu_secret_key
```

### Configuração de Backend

O serviço pode ser configurado para usar diferentes backends em `config.py`:

```python
BACKEND_CONFIG = {
    "fallback_enabled": True,  # Habilita fallback automático
    "preferred_backend": "local",  # "local" ou "api"
    "cache_embeddings": True,  # Cache de embeddings
    "cache_results": True,  # Cache de resultados
    "monitoring": {
        "track_latency": True,
        "track_errors": True,
        "track_cache_hits": True
    }
}
```

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo/services/voice_generator
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Baixe os modelos:
```bash
python download_models.py
```

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite .env com suas configurações
```

5. Inicie o servidor:
```bash
./run.sh
```

## Uso

### Síntese Básica

```bash
curl -X POST "http://localhost:8000/generate-voice" \
  -H "Authorization: Bearer seu_token" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Olá, mundo!",
    "language": "pt",
    "speed": 1.0
  }'
```

### Clonagem de Voz

```bash
curl -X POST "http://localhost:8000/generate-voice" \
  -H "Authorization: Bearer seu_token" \
  -F "text=Olá, mundo!" \
  -F "language=pt" \
  -F "sample=@seu_audio.wav"
```

### Streaming via WebSocket

```python
import asyncio
import websockets
import json

async def tts_stream():
    uri = "ws://localhost:8000/ws/tts"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "text": "Olá, mundo!",
            "language": "pt"
        }))
        
        while True:
            chunk = await websocket.recv()
            if isinstance(chunk, bytes):
                # Processa chunk de áudio
                process_audio(chunk)
            else:
                data = json.loads(chunk)
                if data.get("event") == "end":
                    break

asyncio.run(tts_stream())
```

## Monitoramento

O serviço expõe métricas Prometheus em `/metrics`:

- Latência de geração
- Taxa de erros
- Hits/misses do cache
- Uso de memória GPU
- Tempo de processamento
- **Novo**: Métricas de streaming
- **Novo**: Métricas de validação
- **Novo**: Métricas de I/O

## Tratamento de Erros

O serviço implementa tratamento robusto de erros com classes específicas:

- `VoiceGenerationError`: Erro base
- `BackendError`: Erro no backend
- `APIError`: Erro na API
- `ModelError`: Erro no modelo
- `ValidationError`: Erro de validação
- `AudioProcessingError`: Erro no processamento
- `WebSocketError`: Erro no streaming

## Validação de Dados

Uso de Pydantic para validação robusta:

```python
from schemas import VoiceRequest, VoiceParameters, Language, Emotion

request = VoiceRequest(
    texto="Olá, mundo!",
    parametros=VoiceParameters(
        language=Language.PT,
        speed=1.0,
        emotion=Emotion.HAPPY
    )
)
```

## Contribuindo

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Crie um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes. 