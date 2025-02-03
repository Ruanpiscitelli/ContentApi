# Documentação da API de Geração de Conteúdo

Esta documentação detalha todos os endpoints disponíveis na API de geração de conteúdo, incluindo exemplos de uso.

## Índice
- [Autenticação](#autenticação)
- [Acessando Arquivos Gerados](#acessando-arquivos-gerados)
- [Geração de Imagens](#geração-de-imagens)
- [Geração de Voz](#geração-de-voz)
- [Geração de Vídeo](#geração-de-vídeo)
- [Edição de Vídeo](#edição-de-vídeo)
- [Monitoramento](#monitoramento)

## Acessando Arquivos Gerados

Todos os endpoints de geração retornam uma URL pré-assinada do MinIO que permite acessar o arquivo gerado. A URL é válida por:
- Imagens: 2 dias
- Áudios: 7 dias
- Vídeos: 2 dias

### Exemplo de Resposta (Imagem):
```json
{
    "status": "sucesso",
    "message": "Imagem gerada com sucesso.",
    "minio_url": "https://minio.ruanpiscitelli.com/imagens-geradas/123e4567-e89b-12d3-a456-426614174000.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=..."
}
```

### Exemplo de Resposta (Áudio):
```json
{
    "status": "sucesso",
    "job_id": "voz_1234567890",
    "message": "Áudio gerado com sucesso (duração: 30s).",
    "minio_url": "https://minio.ruanpiscitelli.com/audios-gerados/123e4567-e89b-12d3-a456-426614174000.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=..."
}
```

### Exemplo de Resposta (Vídeo):
```json
{
    "status": "success",
    "video_url": "https://minio.ruanpiscitelli.com/videos-gerados/123e4567-e89b-12d3-a456-426614174000.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=..."
}
```

Para acessar o arquivo:
1. Use a URL retornada no campo `minio_url` ou `video_url`
2. A URL já contém todos os parâmetros de autenticação necessários
3. Você pode baixar o arquivo diretamente ou incorporá-lo em sua aplicação

**Observação**: As URLs são temporárias e expiram após o período especificado. Salve o arquivo localmente se precisar de acesso permanente.

## Autenticação

Todos os endpoints requerem autenticação via token Bearer. Inclua o header `Authorization` em todas as requisições:

```bash
Authorization: Bearer your_api_token_here
```

## Geração de Imagens

### Gerar Imagem

**Endpoint:** `POST /generate-image`

Gera uma imagem usando SDXL ou Flux com suporte a LoRAs e ControlNet.

**Exemplo de Requisição:**
```bash
curl -X POST http://localhost:8001/generate-image \
  -H "Authorization: Bearer your_api_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "cidade futurista durante o pôr do sol",
    "model": "sdxl",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "template_id": "image_template1",
    "loras": [
      {
        "path": "models/loras/cyberpunk_style.safetensors",
        "scale": 0.75
      }
    ]
  }'
```

**Exemplo com ControlNet:**
```bash
curl -X POST http://localhost:8001/generate-image \
  -H "Authorization: Bearer your_api_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "cidade futurista",
    "controlnet": {
      "model_id": "lllyasviel/control_v11p_sd15_canny",
      "image_url": "https://example.com/reference.jpg",
      "preprocessor": "canny",
      "preprocessor_params": {
        "low_threshold": 100,
        "high_threshold": 200
      }
    }
  }'
```

### Verificar Status da Tarefa

**Endpoint:** `GET /task/{task_id}`

Verifica o status de uma tarefa de geração.

```bash
curl -X GET http://localhost:8001/task/123e4567-e89b-12d3-a456-426614174000 \
  -H "Authorization: Bearer your_api_token_here"
```

### Cancelar Tarefa

**Endpoint:** `DELETE /task/{task_id}`

Cancela uma tarefa pendente.

```bash
curl -X DELETE http://localhost:8001/task/123e4567-e89b-12d3-a456-426614174000 \
  -H "Authorization: Bearer your_api_token_here"
```

### Status da Fila

**Endpoint:** `GET /queue/status`

Obtém informações sobre a fila de processamento.

```bash
curl -X GET http://localhost:8001/queue/status \
  -H "Authorization: Bearer your_api_token_here"
```

## Geração de Voz

### Gerar Áudio

**Endpoint:** `POST /generate-voice`

Gera áudio a partir de texto usando Fish Speech.

```bash
curl -X POST http://localhost:8002/generate-voice \
  -H "Authorization: Bearer your_api_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "texto": "Olá, bem-vindo ao meu canal!",
    "template_id": "voice_template_youtube",
    "parametros": {
      "language": "auto",
      "speed": 1.2,
      "pitch": 0.0,
      "energy": 1.2
    }
  }'
```

### Clonar Voz

**Endpoint:** `POST /generate-voice`

Gera áudio usando clonagem de voz.

```bash
curl -X POST http://localhost:8002/generate-voice \
  -H "Authorization: Bearer your_api_token_here" \
  -F "texto=Olá, este é um teste de clonagem de voz" \
  -F "use_voice_clone=true" \
  -F "sample=@/path/to/reference.wav"
```

### Métricas de Voz

**Endpoint:** `GET /metrics/voice`

Obtém métricas do serviço de voz.

```bash
curl -X GET http://localhost:8002/metrics/voice \
  -H "Authorization: Bearer your_api_token_here"
```

## Geração de Vídeo

### Gerar Vídeo

**Endpoint:** `POST /generate-video`

Gera vídeo usando FastHunyuan.

```bash
curl -X POST http://localhost:8003/generate-video \
  -H "Authorization: Bearer your_api_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "um pássaro voando sobre a cidade",
    "template_id": "video_generator_template1",
    "num_frames": 64,
    "fps": 30,
    "width": 1024,
    "height": 576
  }'
```

## Edição de Vídeo

### Editar Vídeo

**Endpoint:** `POST /edit-video`

Aplica edições em um vídeo existente.

```bash
curl -X POST http://localhost:8004/edit-video \
  -H "Authorization: Bearer your_api_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "video_editor_template1",
    "input": "input_video.mp4",
    "operations": [
      {"action": "trim", "start": 5, "end": 20},
      {"action": "overlay", "file": "logo.png", "position": "top-right"}
    ],
    "output": {
      "file": "edited_video.mp4",
      "codec": "libx264"
    }
  }'
```

### Remover Silêncio

**Endpoint:** `POST /cut-silence`

Remove automaticamente períodos de silêncio de um vídeo.

**Parâmetros:**
- `input_url` (string, obrigatório): URL do vídeo de entrada
- `silence_threshold` (float, opcional, default: -35): Limiar de silêncio em dB
  - -30: Corta cliques de mouse e movimentos
  - -35: Corta respiração antes de falar
  - -40 a -50: Cortes quase imperceptíveis
- `min_silence_duration` (float, opcional, default: 1.0): Duração mínima do silêncio em segundos
- `output_format` (string, opcional, default: "mp4"): Formato do vídeo de saída

**Exemplo de Requisição:**
```bash
curl -X POST http://localhost:8004/cut-silence \
  -H "Authorization: Bearer your_api_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "input_url": "https://exemplo.com/video.mp4",
    "silence_threshold": -35,
    "min_silence_duration": 1.0,
    "output_format": "mp4"
  }'
```

**Exemplo de Resposta:**
```json
{
    "status": "success",
    "url": "https://minio.ruanpiscitelli.com/videos-edited/no_silence_uuid.mp4",
    "job_id": "job_uuid",
    "silences_removed": 5
}
```

## Monitoramento

### Healthcheck

**Endpoint:** `GET /healthcheck`

Verifica o status do serviço.

```bash
curl -X GET http://localhost:8001/healthcheck
curl -X GET http://localhost:8002/healthcheck
curl -X GET http://localhost:8003/healthcheck
curl -X GET http://localhost:8004/healthcheck
```

## Códigos de Erro

- `400`: Requisição inválida
- `401`: Não autorizado
- `404`: Recurso não encontrado
- `429`: Limite de requisições excedido
- `500`: Erro interno do servidor

## Limites e Restrições

### Geração de Imagens
- Resolução máxima: 2048x2048
- Formatos suportados: PNG
- Tamanho máximo do prompt: 1000 caracteres

### Geração de Voz
- Duração máxima: 1800 segundos (30 minutos)
- Formatos suportados: WAV
- Idiomas suportados: auto, en, zh, ja, ko, fr, de, ar, es

### Geração de Vídeo
- Resolução máxima: 1280x720
- Duração máxima: 128 frames
- FPS máximo: 30

### Fontes de Entrada para Vídeos
Todos os endpoints que aceitam vídeos como entrada (`/edit-video`, `/cut-silence`, etc.) suportam as seguintes fontes:

1. **URLs HTTP/HTTPS diretas**
   - Links públicos de download
   - CDNs
   - Servidores web
   - Exemplo: `https://exemplo.com/video.mp4`

2. **MinIO**
   - Use o formato: `minio://bucket-name/path/to/video.mp4`
   - Exemplo: `minio://videos-raw/uploads/video.mp4`
   - Acesso direto aos buckets configurados no servidor

3. **Google Drive**
   - Links compartilhados do Google Drive
   - Formatos suportados:
     - `https://drive.google.com/file/d/{FILE_ID}/view`
     - `https://drive.google.com/open?id={FILE_ID}`
   - Exemplo: `https://drive.google.com/file/d/1234567890abcdef/view`

4. **Dropbox**
   - Links compartilhados do Dropbox
   - O sistema converte automaticamente para download direto
   - Exemplo: `https://www.dropbox.com/s/abcdef1234567890/video.mp4`

5. **Supabase Storage**
   - URLs do Supabase Storage
   - Exemplo: `https://supabase.co/storage/v1/object/public/bucket/video.mp4`

6. **URLs Pré-assinadas**
   - URLs temporárias de serviços de armazenamento (S3, MinIO, etc)
   - Exemplo: `https://minio.exemplo.com/bucket/video.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=...`

**Observações Importantes:**
- Para Google Drive e Dropbox, certifique-se que os arquivos estão configurados como "público" ou "qualquer pessoa com o link"
- Para URLs protegidas ou que requerem autenticação, use URLs pré-assinadas
- O tamanho máximo do arquivo é limitado pela configuração do servidor
- Certifique-se que o servidor tenha acesso à fonte do vídeo
- Para arquivos grandes, recomenda-se fazer o upload direto para o MinIO primeiro
- O download é feito em chunks para melhor performance com arquivos grandes

**Exemplos de Uso:**

1. **Com Google Drive:**
```json
{
    "input_url": "https://drive.google.com/file/d/1234567890abcdef/view",
    "silence_threshold": -35,
    "min_silence_duration": 1.0
}
```

2. **Com Dropbox:**
```json
{
    "input_url": "https://www.dropbox.com/s/abcdef1234567890/video.mp4",
    "silence_threshold": -35,
    "min_silence_duration": 1.0
}
```

3. **Com Supabase:**
```json
{
    "input_url": "https://supabase.co/storage/v1/object/public/bucket/video.mp4",
    "silence_threshold": -35,
    "min_silence_duration": 1.0
}
```

## Templates

Os templates estão disponíveis no diretório `templates/` e incluem:
- `image_template1.json`: Template padrão para geração de imagens
- `image_template_ultrarealistic.json`: Template para fotos ultra realistas
- `voice_template1.json`: Template padrão para geração de voz
- `voice_template_youtube.json`: Template otimizado para narrações do YouTube
- `video_generator_template1.json`: Template para geração de vídeos
- `video_editor_template1.json`: Template para edição de vídeos 