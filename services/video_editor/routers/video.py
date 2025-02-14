"""
Router para endpoints de edição de vídeo.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import Dict, Any

from ..models import VideoRequest, VideoEditRequest, SilenceCutRequest
from ..security import verify_bearer_token
from ..pipeline import pipeline
from ..storage import upload_to_minio

router = APIRouter(
    prefix="/video",
    tags=["video"],
    dependencies=[Depends(verify_bearer_token)]
)

@router.post("/edit/basic")
async def edit_video_basic(request: VideoRequest) -> Dict[str, Any]:
    """
    Endpoint para edição básica de vídeo.
    
    Args:
        request: Parâmetros da requisição
        
    Returns:
        Informações do vídeo processado
        
    Raises:
        HTTPException: Se houver erro no processamento
    """
    try:
        # Adiciona tarefa ao pipeline
        task_id = await pipeline.add_task(request)
        
        # Aguarda processamento
        result = await pipeline.wait_for_task(task_id)
        
        if result.error:
            raise HTTPException(
                status_code=500,
                detail=f"Erro no processamento: {result.error}"
            )
            
        return {
            "status": "success",
            "video_url": result.output_url,
            "task_id": task_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro no processamento: {str(e)}"
        )

@router.post("/edit/advanced")
async def edit_video_advanced(request: VideoEditRequest) -> Dict[str, Any]:
    """
    Endpoint para edição avançada de vídeo.
    
    Args:
        request: Parâmetros da requisição
        
    Returns:
        Informações do vídeo processado
        
    Raises:
        HTTPException: Se houver erro no processamento
    """
    try:
        # Valida operações
        for op in request.operations:
            if not pipeline.validate_operation(op):
                raise HTTPException(
                    status_code=400,
                    detail=f"Operação inválida: {op.action}"
                )
                
        # Adiciona tarefa ao pipeline
        task_id = await pipeline.add_task(request)
        
        # Aguarda processamento
        result = await pipeline.wait_for_task(task_id)
        
        if result.error:
            raise HTTPException(
                status_code=500,
                detail=f"Erro no processamento: {result.error}"
            )
            
        return {
            "status": "success",
            "video_url": result.output_url,
            "task_id": task_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro no processamento: {str(e)}"
        )

@router.post("/cut-silence")
async def cut_silence(request: SilenceCutRequest) -> Dict[str, Any]:
    """
    Remove períodos de silêncio do vídeo.
    
    Args:
        request: Parâmetros da requisição
        
    Returns:
        Informações do vídeo processado
        
    Raises:
        HTTPException: Se houver erro no processamento
    """
    try:
        # Adiciona tarefa ao pipeline
        task_id = await pipeline.add_task(request)
        
        # Aguarda processamento
        result = await pipeline.wait_for_task(task_id)
        
        if result.error:
            raise HTTPException(
                status_code=500,
                detail=f"Erro no processamento: {result.error}"
            )
            
        return {
            "status": "success",
            "video_url": result.output_url,
            "task_id": task_id,
            "silence_segments": result.silence_segments
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro no processamento: {str(e)}"
        ) 