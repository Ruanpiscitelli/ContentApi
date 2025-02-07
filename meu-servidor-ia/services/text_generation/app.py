"""
Serviço de Geração de Texto usando vLLM OpenAI API
"""
import os
import time
import logging
import logging.config
from typing import Dict, Any, Optional, List, Union, Literal
import httpx
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, root_validator
import asyncio
from prometheus_client import Counter, Histogram, start_http_server
import redis
import json
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential
from config import (
    LOGGING_CONFIG,
    REDIS_CONFIG,
    GENERATION_LIMITS,
    VLLM_CONFIG,
    RATE_LIMIT
)
from fastapi.middleware.base import BaseHTTPMiddleware
import psutil

# Configuração de logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Configuração do FastAPI
app = FastAPI(
    title="Serviço de Geração de Texto",
    description="API compatível com OpenAI para geração de texto usando vLLM",
    version="1.0.0"
)

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Métricas Prometheus
TEXT_GENERATION_TIME = Histogram(
    'text_generation_seconds', 
    'Tempo de geração de texto',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

TEXT_GENERATION_ERRORS = Counter(
    'text_generation_errors_total',
    'Total de erros na geração de texto'
)

CACHE_HITS = Counter(
    'text_cache_hits_total',
    'Total de hits no cache de texto'
)

CACHE_MISSES = Counter(
    'text_cache_misses_total',
    'Total de misses no cache de texto'
)

VLLM_LATENCY = Histogram(
    'vllm_request_latency_seconds',
    'Latência das requisições ao vLLM'
)

VLLM_ERRORS = Counter(
    'vllm_errors_total',
    'Total de erros nas chamadas ao vLLM',
    ['error_type']
)

MODEL_TOKENS_GENERATED = Counter(
    'model_tokens_generated_total',
    'Total de tokens gerados por modelo',
    ['model']
)

# Cliente Redis
redis_client = redis.Redis.from_url(
    REDIS_CONFIG["url"],
    password=REDIS_CONFIG["password"],
    db=REDIS_CONFIG["db"]
)

# Modelos de dados
class ModelPermission(BaseModel):
    """Permissões do modelo."""
    id: str
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False

class ModelCard(BaseModel):
    """Informações sobre um modelo disponível."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "organization"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = Field(default_factory=list)

class ModelList(BaseModel):
    """Lista de modelos disponíveis."""
    object: str = "list"
    data: List[ModelCard]

class UsageInfo(BaseModel):
    """Informações de uso de tokens."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class CompletionRequest(BaseModel):
    """Modelo de requisição para geração de texto."""
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = Field(default=16, le=GENERATION_LIMITS["max_tokens_output"])
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    best_of: Optional[int] = Field(default=1, ge=1)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    @root_validator
    def validate_all(cls, values):
        """Valida todos os campos."""
        # Valida prompt
        prompt = values.get("prompt")
        if isinstance(prompt, str):
            if len(prompt) > GENERATION_LIMITS["max_prompt_length"]:
                raise ValueError(f"Prompt excede o limite de {GENERATION_LIMITS['max_prompt_length']} caracteres")
        elif isinstance(prompt, list):
            for p in prompt:
                if len(p) > GENERATION_LIMITS["max_prompt_length"]:
                    raise ValueError(f"Prompt excede o limite de {GENERATION_LIMITS['max_prompt_length']} caracteres")
        
        # Valida max_tokens
        max_tokens = values.get("max_tokens")
        if max_tokens and max_tokens > GENERATION_LIMITS["max_tokens_output"]:
            raise ValueError(f"max_tokens não pode exceder {GENERATION_LIMITS['max_tokens_output']}")
        
        # Valida n e best_of
        n = values.get("n", 1)
        best_of = values.get("best_of", 1)
        if best_of < n:
            raise ValueError("best_of deve ser maior ou igual a n")
        
        # Valida stop
        stop = values.get("stop")
        if stop and isinstance(stop, list) and len(stop) > 4:
            raise ValueError("Máximo de 4 sequências de parada permitidas")
        
        return values

class CompletionResponseChoice(BaseModel):
    """Escolha de completação."""
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

class CompletionResponse(BaseModel):
    """Modelo de resposta para geração de texto."""
    id: str = Field(default_factory=lambda: f"cmpl-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:10]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo

class ChatMessage(BaseModel):
    """Mensagem de chat."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    """Requisição de chat completion."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, str]]] = None

    @root_validator
    def validate_all(cls, values):
        """Valida todos os campos."""
        # Valida mensagens
        messages = values.get("messages", [])
        total_length = sum(len(msg.content) for msg in messages)
        if total_length > GENERATION_LIMITS["max_prompt_length"]:
            raise ValueError(f"Total de caracteres das mensagens excede {GENERATION_LIMITS['max_prompt_length']}")
        
        # Valida max_tokens
        max_tokens = values.get("max_tokens")
        if max_tokens and max_tokens > GENERATION_LIMITS["max_tokens_output"]:
            raise ValueError(f"max_tokens não pode exceder {GENERATION_LIMITS['max_tokens_output']}")
        
        # Valida stop
        stop = values.get("stop")
        if stop and isinstance(stop, list) and len(stop) > 4:
            raise ValueError("Máximo de 4 sequências de parada permitidas")
        
        return values

class ChatCompletionResponseChoice(BaseModel):
    """Escolha de resposta do chat."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    """Resposta de chat completion."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:10]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

class EmbeddingRequest(BaseModel):
    """Requisição de embedding."""
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None
    encoding_format: Optional[str] = "float"

class EmbeddingResponse(BaseModel):
    """Resposta de embedding."""
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    status: str
    gpu_memory_usage: float
    loaded_at: float

# Constantes para headers de streaming
STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "text/event-stream",
    "X-Accel-Buffering": "no"
}

# Funções auxiliares
def verify_token(request: Request) -> str:
    """Verifica o token de autenticação."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Token de autenticação ausente ou inválido"
        )
    
    token = auth_header.split(" ")[1]
    if token != os.getenv("API_TOKEN"):
        raise HTTPException(
            status_code=401,
            detail="Token de autenticação inválido"
        )
    
    return token

def get_cache_key(request: Union[CompletionRequest, ChatCompletionRequest]) -> str:
    """Gera chave de cache para a requisição."""
    if isinstance(request, CompletionRequest):
        data = {
            "type": "completion",
            "model": request.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "stop": request.stop
        }
    else:
        data = {
            "type": "chat",
            "model": request.model,
            "messages": [msg.dict() for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "stop": request.stop
        }
    return f"vllm:{hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()}"

def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Formata mensagens de chat para prompt."""
    formatted = []
    for msg in messages:
        if msg.role == "system":
            formatted.append(f"System: {msg.content}")
        elif msg.role == "user":
            formatted.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            formatted.append(f"Assistant: {msg.content}")
        elif msg.role == "tool":
            formatted.append(f"Tool ({msg.name}): {msg.content}")
    return "\n".join(formatted)

async def stream_completion(response: Dict[str, Any], is_chat: bool = False):
    """Função para streaming de respostas."""
    try:
        for choice in response["choices"]:
            if is_chat:
                delta = {
                    "id": f"chatcmpl-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:10]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": response["model"],
                    "choices": [{
                        "index": choice["index"],
                        "delta": {"content": choice["text"]},
                        "finish_reason": None
                    }]
                }
            else:
                delta = {
                    "id": f"cmpl-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:10]}",
                    "object": "text_completion.chunk",
                    "created": int(time.time()),
                    "model": response["model"],
                    "choices": [{
                        "text": choice["text"],
                        "index": choice["index"],
                        "logprobs": None,
                        "finish_reason": None
                    }]
                }
            yield f"data: {json.dumps(delta)}\n\n"
        
        # Envia o finish_reason no último chunk
        if is_chat:
            yield f"data: {json.dumps({
                'id': f"chatcmpl-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:10]}",
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': response['model'],
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'stop'
                }]
            })}\n\n"
        else:
            yield f"data: {json.dumps({
                'id': f"cmpl-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:10]}",
                'object': 'text_completion.chunk',
                'created': int(time.time()),
                'model': response['model'],
                'choices': [{
                    'text': '',
                    'index': 0,
                    'logprobs': None,
                    'finish_reason': 'stop'
                }]
            })}\n\n"
        
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Erro no streaming: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def call_vllm_api(endpoint: str, method: str = "GET", json_data: Optional[Dict] = None) -> Dict[str, Any]:
    """Função genérica para chamar a API do vLLM com retry."""
    vllm_base = os.getenv("VLLM_ENDPOINT", "http://vllm:8000").rstrip("/")
    url = f"{vllm_base}/{endpoint.lstrip('/')}"
    
    async with httpx.AsyncClient() as client:
        try:
            start_time = time.time()
            if method == "GET":
                response = await client.get(url, timeout=GENERATION_LIMITS["timeout"])
            else:
                response = await client.post(
                    url,
                    json=json_data,
                    timeout=GENERATION_LIMITS["timeout"]
                )
            
            VLLM_LATENCY.observe(time.time() - start_time)
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            VLLM_ERRORS.labels(error_type="http").inc()
            logger.error(f"Erro HTTP na chamada vLLM: {str(e)}")
            raise HTTPException(
                status_code=response.status_code if response else 500,
                detail=f"Erro no serviço vLLM: {str(e)}"
            )
        except Exception as e:
            VLLM_ERRORS.labels(error_type="other").inc()
            logger.error(f"Erro na chamada vLLM: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Erro interno no serviço vLLM: {str(e)}"
            )

async def check_vllm_health() -> bool:
    """Verifica a saúde do serviço vLLM."""
    try:
        await call_vllm_api("health")
        return True
    except Exception:
        return False

# Middlewares
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.last_request_time = {}
        
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Verifica endpoints que precisam de rate limiting
        if request.url.path in ["/v1/completions", "/v1/chat/completions"]:
            try:
                body = await request.json()
                model = body.get("model")
                if model in VLLM_CONFIG["available_models"]:
                    limit = RATE_LIMIT["per_model_limits"].get(model, RATE_LIMIT["requests_per_minute"])
                    
                    # Verifica rate limit
                    if client_ip in self.last_request_time:
                        time_passed = current_time - self.last_request_time[client_ip]
                        if time_passed < (60 / limit):
                            raise HTTPException(
                                status_code=429,
                                detail=f"Rate limit excedido para o modelo {model}"
                            )
                    
                    self.last_request_time[client_ip] = current_time
            except json.JSONDecodeError:
                pass
        
        response = await call_next(request)
        return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para logging de requisições."""
    start_time = time.time()
    
    # Log da requisição
    logger.info(f"Request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log da resposta
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"Status: {response.status_code} "
            f"Time: {process_time:.2f}s"
        )
        
        return response
    except Exception as e:
        logger.error(
            f"Error: {request.method} {request.url.path} "
            f"Error: {str(e)}"
        )
        raise

# Adiciona middlewares
app.add_middleware(RateLimitMiddleware)

# Endpoints
@app.get("/v1/models", response_model=ModelList)
async def list_models(token: str = Depends(verify_token)):
    """Lista todos os modelos disponíveis."""
    try:
        models = []
        for model_id in VLLM_CONFIG["available_models"]:
            models.append(ModelCard(id=model_id))
        return ModelList(data=models)
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao listar modelos: {str(e)}"
        )

@app.get("/v1/models/{model_id}")
async def get_model_info(
    model_id: str,
    token: str = Depends(verify_token)
):
    """Obtém informações sobre um modelo específico."""
    try:
        if model_id not in VLLM_CONFIG["available_models"]:
            raise HTTPException(
                status_code=404,
                detail=f"Modelo {model_id} não encontrado"
            )
        
        model_info = await call_vllm_api(f"v1/models/{model_id}")
        return ModelCard(
            id=model_id,
            permission=[ModelPermission(id=f"{model_id}-permission")]
        )
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao obter informações do modelo: {str(e)}"
        )

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Gera texto a partir do prompt fornecido usando o vLLM."""
    start_time = time.time()
    
    try:
        # Validar modelo
        if request.model not in VLLM_CONFIG["available_models"]:
            raise HTTPException(
                status_code=404,
                detail=f"Modelo {request.model} não encontrado"
            )
        
        # Verifica cache se não for streaming
        if not request.stream:
            cache_key = get_cache_key(request)
            cached_response = redis_client.get(cache_key)
            
            if cached_response:
                CACHE_HITS.inc()
                return CompletionResponse(**json.loads(cached_response))
            
            CACHE_MISSES.inc()
        
        # Prepara request para vLLM
        vllm_request = {
            "model": request.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.n,
            "stream": request.stream,
            "stop": request.stop,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "best_of": request.best_of
        }
        
        # Remove campos None
        vllm_request = {k: v for k, v in vllm_request.items() if v is not None}
        
        # Chama vLLM API
        vllm_response = await call_vllm_api(
            "v1/completions",
            method="POST",
            json_data=vllm_request
        )
        
        # Atualiza métricas
        process_time = time.time() - start_time
        TEXT_GENERATION_TIME.observe(process_time)
        MODEL_TOKENS_GENERATED.labels(model=request.model).inc(
            vllm_response["usage"]["completion_tokens"]
        )
        
        # Stream response se necessário
        if request.stream:
            return StreamingResponse(
                stream_completion(vllm_response),
                media_type="text/event-stream",
                headers=STREAM_HEADERS
            )
        
        # Prepara resposta normal
        choices = [
            CompletionResponseChoice(
                text=choice["text"],
                index=choice["index"],
                logprobs=choice.get("logprobs"),
                finish_reason=choice.get("finish_reason")
            )
            for choice in vllm_response["choices"]
        ]
        
        response = CompletionResponse(
            model=request.model,
            choices=choices,
            usage=UsageInfo(**vllm_response["usage"])
        )
        
        # Cache da resposta
        if not request.stream:
            background_tasks.add_task(
                redis_client.setex,
                cache_key,
                REDIS_CONFIG["ttl"],
                json.dumps(response.dict())
            )
        
        return response
        
    except Exception as e:
        TEXT_GENERATION_ERRORS.inc()
        logger.error(f"Erro na geração de texto: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro na geração de texto: {str(e)}"
        )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Gera resposta de chat usando o vLLM."""
    start_time = time.time()
    
    try:
        # Validar modelo
        if request.model not in VLLM_CONFIG["available_models"]:
            raise HTTPException(
                status_code=404,
                detail=f"Modelo {request.model} não encontrado"
            )
        
        # Verifica cache se não for streaming
        if not request.stream:
            cache_key = get_cache_key(request)
            cached_response = redis_client.get(cache_key)
            
            if cached_response:
                CACHE_HITS.inc()
                return ChatCompletionResponse(**json.loads(cached_response))
            
            CACHE_MISSES.inc()
        
        # Formata mensagens para prompt
        prompt = format_chat_prompt(request.messages)
        
        # Prepara request para vLLM
        vllm_request = {
            "model": request.model,
            "prompt": prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.n,
            "stream": request.stream,
            "stop": request.stop,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty
        }
        
        # Remove campos None
        vllm_request = {k: v for k, v in vllm_request.items() if v is not None}
        
        # Chama vLLM API
        vllm_response = await call_vllm_api(
            "v1/completions",
            method="POST",
            json_data=vllm_request
        )
        
        # Atualiza métricas
        process_time = time.time() - start_time
        TEXT_GENERATION_TIME.observe(process_time)
        MODEL_TOKENS_GENERATED.labels(model=request.model).inc(
            vllm_response["usage"]["completion_tokens"]
        )
        
        # Stream response se necessário
        if request.stream:
            return StreamingResponse(
                stream_completion(vllm_response, is_chat=True),
                media_type="text/event-stream",
                headers=STREAM_HEADERS
            )
        
        # Prepara resposta normal
        choices = [
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(
                    role="assistant",
                    content=choice["text"]
                ),
                finish_reason=choice.get("finish_reason")
            )
            for i, choice in enumerate(vllm_response["choices"])
        ]
        
        response = ChatCompletionResponse(
            model=request.model,
            choices=choices,
            usage=UsageInfo(**vllm_response["usage"])
        )
        
        # Cache da resposta
        if not request.stream:
            background_tasks.add_task(
                redis_client.setex,
                cache_key,
                REDIS_CONFIG["ttl"],
                json.dumps(response.dict())
            )
        
        return response
        
    except Exception as e:
        TEXT_GENERATION_ERRORS.inc()
        logger.error(f"Erro na geração de chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro na geração de chat: {str(e)}"
        )

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    token: str = Depends(verify_token)
):
    """Gera embeddings usando o vLLM."""
    try:
        # Validar modelo
        if request.model not in VLLM_CONFIG["available_models"]:
            raise HTTPException(
                status_code=404,
                detail=f"Modelo {request.model} não encontrado"
            )
        
        # Prepara request para vLLM
        vllm_request = {
            "model": request.model,
            "input": request.input,
            "encoding_format": request.encoding_format
        }
        
        # Chama vLLM API
        vllm_response = await call_vllm_api(
            "v1/embeddings",
            method="POST",
            json_data=vllm_request
        )
        
        return EmbeddingResponse(
            data=vllm_response["data"],
            model=request.model,
            usage=UsageInfo(**vllm_response["usage"])
        )
        
    except Exception as e:
        logger.error(f"Erro na geração de embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro na geração de embeddings: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde."""
    try:
        vllm_status = await check_vllm_health()
        redis_status = redis_client.ping()
        
        return {
            "status": "ok" if (vllm_status and redis_status) else "degraded",
            "service": "text-generator",
            "components": {
                "vllm": "ok" if vllm_status else "error",
                "redis": "ok" if redis_status else "error"
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Erro no health check: {str(e)}")
        return {
            "status": "error",
            "service": "text-generator",
            "error": str(e),
            "timestamp": time.time()
        }

@app.on_event("startup")
async def startup_event():
    """Evento de inicialização."""
    try:
        logger.info("Iniciando serviço de geração de texto...")
        
        # Verifica conexão com Redis
        try:
            redis_client.ping()
            logger.info("✓ Conexão com Redis estabelecida")
        except Exception as e:
            logger.error(f"✗ Erro ao conectar ao Redis: {str(e)}")
            raise e
        
        # Verifica conexão com vLLM
        try:
            vllm_status = await check_vllm_health()
            if not vllm_status:
                raise Exception("Serviço não respondeu")
            logger.info("✓ Conexão com vLLM estabelecida")
        except Exception as e:
            logger.error(f"✗ Erro ao conectar ao vLLM: {str(e)}")
            raise e
        
        # Verifica modelos configurados
        try:
            models_response = await call_vllm_api("v1/models")
            available_models = set(model["id"] for model in models_response.get("data", []))
            configured_models = set(VLLM_CONFIG["available_models"])
            
            if not configured_models:
                raise Exception("Nenhum modelo configurado")
            
            if not configured_models.issubset(available_models):
                missing_models = configured_models - available_models
                raise Exception(f"Modelos não disponíveis: {missing_models}")
            
            logger.info(f"✓ {len(configured_models)} modelos disponíveis: {', '.join(configured_models)}")
        except Exception as e:
            logger.error(f"✗ Erro ao verificar modelos: {str(e)}")
            raise e
        
        # Verifica recursos do sistema
        try:
            memory = psutil.virtual_memory()
            logger.info(f"✓ Memória disponível: {memory.available / (1024**3):.1f}GB de {memory.total / (1024**3):.1f}GB ({memory.percent}% em uso)")
            
            try:
                import py3nvml.py3nvml as nvml
                nvml.nvmlInit()
                device_count = nvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    name = nvml.nvmlDeviceGetName(handle)
                    logger.info(f"✓ GPU {i} ({name.decode()}): {info.free / (1024**3):.1f}GB livre de {info.total / (1024**3):.1f}GB")
            except Exception as e:
                logger.warning(f"ℹ GPU não detectada: {str(e)}")
        except Exception as e:
            logger.error(f"✗ Erro ao verificar recursos: {str(e)}")
            raise e
        
        logger.info("✓ Serviço iniciado com sucesso!")
        
    except Exception as e:
        logger.error(f"✗ Erro fatal na inicialização: {str(e)}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Evento de desligamento."""
    try:
        logger.info("Iniciando desligamento do serviço...")
        
        # Fecha conexão com Redis
        try:
            redis_client.close()
            logger.info("✓ Conexão com Redis fechada")
        except Exception as e:
            logger.error(f"✗ Erro ao fechar conexão com Redis: {str(e)}")
        
        # Aguarda requisições pendentes
        try:
            await asyncio.sleep(1)
            logger.info("✓ Requisições pendentes finalizadas")
        except Exception as e:
            logger.error(f"✗ Erro ao aguardar requisições: {str(e)}")
        
        # Libera recursos da GPU
        try:
            import py3nvml.py3nvml as nvml
            nvml.nvmlShutdown()
            logger.info("✓ Recursos da GPU liberados")
        except Exception as e:
            logger.warning(f"ℹ Erro ao liberar GPU: {str(e)}")
        
        logger.info("✓ Serviço finalizado com sucesso!")
        
    except Exception as e:
        logger.error(f"✗ Erro no desligamento: {str(e)}")
        # Não re-levanta a exceção para permitir shutdown limpo

if __name__ == "__main__":
    # Iniciar servidor de métricas
    start_http_server(int(os.getenv("METRICS_PORT", 8001)))
    
    # Iniciar FastAPI
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000))
    )
