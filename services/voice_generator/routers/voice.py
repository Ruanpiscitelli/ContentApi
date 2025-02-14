"""
Router para endpoints de geração de voz.
"""
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from typing import Dict, Any, Optional
from pydantic import BaseModel

from ..schemas import VoiceRequest
from ..backends import VoiceBackendManager
from ..security import verify_bearer_token

router = APIRouter(
    prefix="/voice",
    tags=["voice"],
    dependencies=[Depends(verify_bearer_token)]
)

@router.post("/generate")
async def generate_voice(
    request: VoiceRequest,
    sample: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    """
    Gera áudio a partir de texto.
    
    Args:
        request: Parâmetros da requisição
        sample: Arquivo de áudio opcional para clonagem de voz
        
    Returns:
        URL do áudio gerado e informações adicionais
    """
    try:
        # Processa a requisição
        audio_data = await VoiceBackendManager().generate(
            text=request.texto,
            clone_audio=await sample.read() if sample else None,
            **request.parametros.dict()
        )
        
        # Upload do resultado
        url = upload_to_minio(audio_data, "audio.wav")
        
        return {
            "status": "success",
            "url": url,
            "duration": get_audio_duration(audio_data)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na geração: {str(e)}"
        )

@router.get("/voices")
async def list_voices() -> Dict[str, Any]:
    """Lista vozes disponíveis."""
    try:
        voices = await VoiceBackendManager().list_voices()
        return {
            "voices": voices
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao listar vozes: {str(e)}"
        )

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Verifica saúde do serviço."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    } 