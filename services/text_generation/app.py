"""
API principal do serviço de geração de texto.
"""
import os
import time
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import start_http_server
from vllm import SamplingParams
from pydantic import BaseModel
import torch
from transformers import pipeline

from config import API_CONFIG, MODEL_CONFIG
from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    UsageInfo,
    ErrorResponse
)
from security import security_handler
from validation import input_sanitizer, RequestValidator
from metrics import (
    REQUEST_COUNTER,
    REQUEST_DURATION,
    TOKENS_GENERATED,
    PROMPT_TOKENS,
    ERROR_COUNTER
)
from resource_manager import ResourceManager
from model_manager import ModelManager
from function_calling import function_registry, function_caller
from routers import text

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Cria aplicação FastAPI
app = FastAPI(
    title="Serviço de Geração de Texto",
    description="API compatível com OpenAI para geração de texto usando vLLM",
    version="1.0.0"
)

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors"]["origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Inicializa gerenciadores
resource_manager = ResourceManager()
model_manager = ModelManager(resource_manager)

# Inclui routers
app.include_router(text.router)

class TextRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7

@app.on_event("startup")
async def startup():
    """Inicializa serviço."""
    # Inicia servidor Prometheus
    start_http_server(API_CONFIG["metrics_port"])
    
    # Inicia handlers
    security_handler.start()
    model_manager.start()
    
    # Otimiza GPUs
    await resource_manager.optimize_gpu_settings()
    
    # Inicia monitoramento
    asyncio.create_task(resource_manager.monitor_resources())
    
    logger.info("Serviço iniciado com sucesso")

@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "Text Generation Service"}

@app.get("/v1/models")
async def list_models(
    token_data: dict = Depends(security_handler.verify_token)
) -> List[Dict[str, Any]]:
    """
    Lista modelos disponíveis.
    
    Args:
        token_data: Dados do token de autenticação
        
    Returns:
        Lista de modelos
    """
    return model_manager.list_models()

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
    token_data: dict = Depends(security_handler.verify_token)
) -> ChatCompletionResponse:
    """
    Gera completions para chat.
    
    Args:
        request: Requisição de completion
        raw_request: Requisição HTTP bruta
        token_data: Dados do token de autenticação
        
    Returns:
        Resposta com texto gerado
    """
    start_time = time.time()
    
    try:
        # Verifica rate limits
        remaining, reset = await security_handler.check_rate_limits(
            token_data["sub"],
            raw_request.client.host
        )
        
        # Sanitiza entrada
        sanitized_data = input_sanitizer.sanitize_json(request.dict())
        
        # Valida request
        validated_data = RequestValidator(**sanitized_data)
        
        # Prepara prompt
        if isinstance(validated_data.prompt, str):
            prompt = validated_data.prompt
        else:
            # Formata mensagens de chat
            prompt = ""
            for msg in validated_data.prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
                else:
                    prompt += f"Human: {content}\n\n"
            
            prompt += "Assistant:"
        
        # Gera texto
        generated_text, usage = await model_manager.generate(
            validated_data.model,
            prompt,
            {
                "temperature": validated_data.temperature,
                "top_p": validated_data.top_p,
                "n": validated_data.n,
                "max_tokens": validated_data.max_tokens,
                "stop": validated_data.stop,
                "presence_penalty": validated_data.presence_penalty,
                "frequency_penalty": validated_data.frequency_penalty
            }
        )
        
        # Processa chamadas de função
        function_calls = []
        if validated_data.functions:
            import ast
            from pydantic import create_model, ValidationError
            
            # Gera respostas únicas para cada choice
            for i in range(validated_data.n or 1):
                try:
                    for func in validated_data.functions:
                        # Procura chamadas de função usando regex simples
                        import re
                        pattern = rf'{func["name"]}\s*\([^)]+\)'
                        matches = re.finditer(pattern, generated_text)
                        
                        for match in matches:
                            try:
                                # Parse usando AST
                                func_call = ast.parse(match.group(0)).body[0].value
                                if not isinstance(func_call, ast.Call):
                                    continue
                                    
                                # Extrai argumentos
                                args = {}
                                for kw in func_call.keywords:
                                    # Avalia valor literal de forma segura
                                    try:
                                        value = ast.literal_eval(kw.value)
                                        args[kw.arg] = value
                                    except (ValueError, SyntaxError):
                                        continue
                                
                                # Cria modelo Pydantic dinâmico para validação
                                if "parameters" in func:
                                    fields = {
                                        p["name"]: (
                                            p.get("type", str),
                                            ... if p.get("required", False) else None
                                        )
                                        for p in func["parameters"]
                                    }
                                    
                                    ValidatorModel = create_model(
                                        f"{func['name']}Args",
                                        **fields
                                    )
                                    
                                    # Valida argumentos
                                    try:
                                        validated_args = ValidatorModel(**args).dict()
                                    except ValidationError as e:
                                        logger.warning(
                                            f"Validação falhou para {func['name']}: {e}"
                                        )
                                        continue
                                else:
                                    validated_args = args
                                
                                # Chama função
                                try:
                                    result = await function_caller.call_function(
                                        func["name"],
                                        validated_args
                                    )
                                    
                                    function_calls.append({
                                        "name": func["name"],
                                        "arguments": validated_args,
                                        "result": result
                                    })
                                    break
                                    
                                except Exception as e:
                                    logger.error(
                                        f"Erro ao executar {func['name']}: {e}"
                                    )
                                    ERROR_COUNTER.labels(
                                        type="function_execution_error",
                                        model=validated_data.model
                                    ).inc()
                                    
                            except SyntaxError:
                                continue
                                
                except Exception as e:
                    logger.error(f"Erro ao processar função: {e}")
                    ERROR_COUNTER.labels(
                        type="function_parsing_error",
                        model=validated_data.model
                    ).inc()
        
        # Prepara resposta com choices únicos
        choices = []
        for i in range(validated_data.n or 1):
            # Gera texto único para cada choice se necessário
            if i > 0:
                generated_text, _ = await model_manager.generate(
                    validated_data.model,
                    prompt,
                    {
                        "temperature": validated_data.temperature,
                        "top_p": validated_data.top_p,
                        "max_tokens": validated_data.max_tokens,
                        "stop": validated_data.stop,
                        "presence_penalty": validated_data.presence_penalty,
                        "frequency_penalty": validated_data.frequency_penalty
                    }
                )
            
            choice = ChatCompletionResponseChoice(
                index=i,
                message={
                    "role": "assistant",
                    "content": generated_text
                }
            )
            
            # Adiciona function call se disponível para este choice
            if i < len(function_calls):
                choice.message["function_call"] = function_calls[i]
            
            choices.append(choice)
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time()*1000)}",
            object="chat.completion",
            created=int(time.time()),
            model=validated_data.model,
            choices=choices,
            usage=UsageInfo(**usage)
        )
        
        # Registra métricas
        duration = time.time() - start_time
        REQUEST_COUNTER.labels(
            status="success",
            model=validated_data.model
        ).inc()
        REQUEST_DURATION.labels(model=validated_data.model).observe(duration)
        TOKENS_GENERATED.labels(model=validated_data.model).inc(
            usage["completion_tokens"]
        )
        PROMPT_TOKENS.labels(model=validated_data.model).observe(
            usage["prompt_tokens"]
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")
        ERROR_COUNTER.labels(
            type=type(e).__name__,
            model=getattr(validated_data, "model", "unknown")
        ).inc()
        
        if isinstance(e, HTTPException):
            raise
            
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/v1/chat/completions/stream")
async def chat_completions_stream(
    request: ChatCompletionRequest,
    raw_request: Request,
    token_data: dict = Depends(security_handler.verify_token)
) -> StreamingResponse:
    """
    Gera completions para chat em streaming.
    
    Args:
        request: Requisição de completion
        raw_request: Requisição HTTP bruta
        token_data: Dados do token de autenticação
        
    Returns:
        Stream de respostas
    """
    async def generate():
        try:
            # Verifica rate limits
            remaining, reset = await security_handler.check_rate_limits(
                token_data["sub"],
                raw_request.client.host
            )
            
            # Sanitiza e valida entrada
            sanitized_data = input_sanitizer.sanitize_json(request.dict())
            validated_data = RequestValidator(**sanitized_data)
            
            # Prepara prompt
            if isinstance(validated_data.prompt, str):
                prompt = validated_data.prompt
            else:
                prompt = ""
                for msg in validated_data.prompt:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        prompt += f"System: {content}\n\n"
                    elif role == "assistant":
                        prompt += f"Assistant: {content}\n\n"
                    else:
                        prompt += f"Human: {content}\n\n"
                
                prompt += "Assistant:"
            
            # Carrega modelo
            info = await model_manager.load_model(validated_data.model)
            
            # Configura parâmetros
            params = SamplingParams(
                temperature=validated_data.temperature or 1.0,
                top_p=validated_data.top_p or 1.0,
                max_tokens=validated_data.max_tokens or 100,
                stop=validated_data.stop,
                presence_penalty=validated_data.presence_penalty or 0.0,
                frequency_penalty=validated_data.frequency_penalty or 0.0
            )
            
            # Gera texto em streaming
            async for output in info.engine.generate(prompt, params, stream=True):
                chunk = output.outputs[0].text
                
                # Cria resposta
                response = {
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": validated_data.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": None
                        }
                    ]
                }
                
                yield f"data: {json.dumps(response)}\n\n"
            
            # Envia mensagem final
            response["choices"][0]["finish_reason"] = "stop"
            yield f"data: {json.dumps(response)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Erro no streaming: {e}")
            ERROR_COUNTER.labels(
                type=type(e).__name__,
                model=getattr(validated_data, "model", "unknown")
            ).inc()
            
            error_response = ErrorResponse(
                error={
                    "message": str(e),
                    "type": type(e).__name__,
                    "param": None,
                    "code": None
                }
            )
            
            yield f"data: {json.dumps(error_response.dict())}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.get("/v1/functions")
async def list_functions(
    token_data: dict = Depends(security_handler.verify_token)
) -> List[Dict[str, Any]]:
    """
    Lista funções disponíveis.
    
    Args:
        token_data: Dados do token de autenticação
        
    Returns:
        Lista de funções
    """
    return function_registry.list_functions()

@app.post("/v1/functions/{name}")
async def call_function(
    name: str,
    arguments: Dict[str, Any],
    token_data: dict = Depends(security_handler.verify_token)
) -> Dict[str, Any]:
    """
    Chama uma função.
    
    Args:
        name: Nome da função
        arguments: Argumentos da função
        token_data: Dados do token de autenticação
        
    Returns:
        Resultado da função
    """
    return await function_caller.call_function(name, arguments)

@app.post("/generate")
async def generate_text(request: TextRequest):
    try:
        generator = pipeline('text-generation', model='gpt2')
        result = generator(
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        return {"generated_text": result[0]['generated_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
