"""
Router para endpoints de geração de vídeo.
"""
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from ..models import VideoRequest, VideoEditRequest, SilenceCutRequest
from ..pipeline import VideoPipeline
from ..security import verify_bearer_token
from ..storage import upload_to_minio

router = APIRouter(
    prefix="/video",
    tags=["video"],
    dependencies=[Depends(verify_bearer_token)]
)

@router.post("/generate")
async def generate_video(request: VideoRequest) -> Dict[str, Any]:
    """
    Gera um vídeo a partir de cenas e elementos.
    
    Args:
        request: Parâmetros da requisição incluindo cenas e configurações
        
    Returns:
        URL do vídeo gerado e informações adicionais
    """
    try:
        # Adiciona tarefa ao pipeline
        task_id = await VideoPipeline().add_task(request)
        
        # Aguarda processamento
        result = await VideoPipeline().get_task_status(task_id)
        
        if result.error:
            raise HTTPException(
                status_code=500,
                detail=f"Erro no processamento: {result.error}"
            )
            
        return {
            "status": "success",
            "url": result.output_url,
            "task_id": task_id,
            "duration": result.duration
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na geração: {str(e)}"
        )

@router.post("/edit")
async def edit_video(request: VideoEditRequest) -> Dict[str, Any]:
    """
    Edita um vídeo existente.
    
    Args:
        request: Instruções de edição
        
    Returns:
        URL do vídeo editado e informações adicionais
    """
    try:
        # Adiciona tarefa ao pipeline
        task_id = await VideoPipeline().add_task(request)
        
        # Aguarda processamento
        result = await VideoPipeline().get_task_status(task_id)
        
        if result.error:
            raise HTTPException(
                status_code=500,
                detail=f"Erro no processamento: {result.error}"
            )
            
        return {
            "status": "success",
            "url": result.output_url,
            "task_id": task_id,
            "duration": result.duration
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na edição: {str(e)}"
        )

@router.post("/cut-silence")
async def cut_silence(request: SilenceCutRequest) -> Dict[str, Any]:
    """
    Remove períodos de silêncio do vídeo.
    
    Args:
        request: Parâmetros para detecção e remoção de silêncio
        
    Returns:
        URL do vídeo processado e informações dos segmentos removidos
    """
    try:
        # Adiciona tarefa ao pipeline
        task_id = await VideoPipeline().add_task(request)
        
        # Aguarda processamento
        result = await VideoPipeline().get_task_status(task_id)
        
        if result.error:
            raise HTTPException(
                status_code=500,
                detail=f"Erro no processamento: {result.error}"
            )
            
        return {
            "status": "success",
            "url": result.output_url,
            "task_id": task_id,
            "silence_segments": result.silence_segments,
            "duration": result.duration
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar silêncio: {str(e)}"
        )

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Obtém status de uma tarefa.
    
    Args:
        task_id: ID da tarefa
        
    Returns:
        Status atual da tarefa
    """
    try:
        result = await VideoPipeline().get_task_status(task_id)
        return {
            "status": result.status,
            "progress": result.progress,
            "error": result.error
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao obter status: {str(e)}"
        )

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Verifica saúde do serviço."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "gpu_available": torch.cuda.is_available()
    } 