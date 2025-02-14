"""
Router para endpoints de geração de texto.
Mantém compatibilidade com a API OpenAI.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any, List, Optional, Union
from fastapi.responses import StreamingResponse
import time
import json

from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    UsageInfo,
    ErrorResponse
)
from ..security import security_handler
from ..validation import input_sanitizer, RequestValidator
from ..metrics import (
    REQUEST_COUNTER,
    REQUEST_DURATION,
    TOKENS_GENERATED,
    PROMPT_TOKENS,
    ERROR_COUNTER
)
from ..model_manager import model_manager
from ..function_calling import function_registry, function_caller

router = APIRouter(
    prefix="/v1",
    tags=["text"]
)

@router.get("/models")
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

@router.post("/chat/completions")
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
        function_call = None
        if validated_data.functions:
            # Verifica se tem chamada de função no texto
            for func in validated_data.functions:
                if func["name"] in generated_text:
                    try:
                        # Extrai argumentos do texto
                        start = generated_text.find(func["name"]) + len(func["name"])
                        end = generated_text.find(")", start)
                        args_str = generated_text[start:end].strip("()")
                        
                        # Converte para dict
                        args = {}
                        for arg in args_str.split(","):
                            key, value = arg.split("=")
                            args[key.strip()] = value.strip().strip('"\'')
                        
                        # Chama função
                        result = await function_caller.call_function(
                            func["name"],
                            args
                        )
                        
                        function_call = {
                            "name": func["name"],
                            "arguments": args,
                            "result": result
                        }
                        break
                        
                    except Exception as e:
                        logger.error(f"Erro ao processar chamada de função: {e}")
                        ERROR_COUNTER.labels(
                            type="function_call_error",
                            model=validated_data.model
                        ).inc()
        
        # Prepara resposta
        choices = []
        for i in range(validated_data.n or 1):
            choice = ChatCompletionResponseChoice(
                index=i,
                message={
                    "role": "assistant",
                    "content": generated_text
                }
            )
            
            if function_call:
                choice.message["function_call"] = function_call
            
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

@router.post("/chat/completions/stream")
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

@router.get("/functions")
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

@router.post("/functions/{name}")
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