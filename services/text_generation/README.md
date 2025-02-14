# Serviço de Geração de Texto

API compatível com OpenAI para geração de texto usando vLLM.

## Características

- Compatível com a API OpenAI
- Suporte a múltiplos modelos
- Processamento em batch otimizado
- Cache hierárquico
- Rate limiting
- Monitoramento via logs
- Segurança com JWT
- Suporte a function calling
- Streaming de respostas
- Tensor parallelism para múltiplas GPUs

## Requisitos

- Python 3.8+
- CUDA 11.7+
- 4x RTX 4090 (ou GPUs similares)
- 64GB+ RAM
- Ubuntu 22.04 ou superior

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/meu-servidor-ia.git
cd meu-servidor-ia/services/text_generation
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite .env com suas configurações
```

## Configuração

O serviço pode ser configurado através do arquivo `config.py` ou variáveis de ambiente:

- `API_HOST`: Host da API (default: 0.0.0.0)
- `API_PORT`: Porta da API (default: 8000)
- `JWT_SECRET_KEY`: Chave secreta para JWT
- `CORS_ORIGINS`: Origens permitidas para CORS

## Uso

1. Inicie o serviço:
```bash
python app.py
```

2. Acesse a documentação da API:
```
http://localhost:8000/docs
```

## Endpoints

### Chat Completions

- `POST /v1/chat/completions`: Gera texto a partir de um prompt
- `POST /v1/chat/completions/stream`: Gera texto em streaming

### Modelos

- `GET /v1/models`: Lista modelos disponíveis

### Funções

- `GET /v1/functions`: Lista funções disponíveis
- `POST /v1/functions/{name}`: Chama uma função

## Monitoramento

O serviço registra métricas importantes nos logs:

- Total de requisições
- Duração das requisições
- Total de tokens gerados
- Hits/misses no cache
- Uso de memória GPU
- Total de erros

## Segurança

O serviço implementa:

- Autenticação JWT
- Rate limiting por IP e token
- Sanitização de entrada
- Validação de parâmetros
- CORS configurável

## Cache

Sistema de cache hierárquico com 3 níveis:

1. Cache em memória (L1)
2. Redis (L2)
3. Disco (L3)

## Otimizações

- Tensor parallelism para múltiplas GPUs
- Processamento em batch otimizado
- Prefetch de modelos
- Compressão de cache
- Cleanup automático

## Desenvolvimento

1. Instale dependências de desenvolvimento:
```bash
pip install -r requirements-dev.txt
```

2. Execute testes:
```bash
pytest
```

3. Verifique estilo:
```bash
flake8
black .
isort .
```

## Contribuindo

1. Fork o repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

MIT 