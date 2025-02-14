"""
Router para endpoints de geração de imagem.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from ..models import ImageRequest, ControlNetConfig, LoRAConfig, VAEConfig
from ..pipeline import generate_image_with_pipeline, load_pipeline
from ..security import verify_bearer_token
from ..storage import upload_to_minio

router = APIRouter(
    prefix="/image",
    tags=["image"],
    dependencies=[Depends(verify_bearer_token)]
)

@router.post("/generate")
async def generate_image(request: ImageRequest) -> Dict[str, Any]:
    """
    Gera uma imagem usando Stable Diffusion.
    
    Args:
        request: Parâmetros da requisição incluindo prompt e configurações
        
    Returns:
        URL da imagem gerada e informações adicionais
    """
    try:
        # Carrega pipeline
        pipeline = await load_pipeline(request.model)
        
        # Gera imagem
        image = await generate_image_with_pipeline(request, pipeline)
        
        # Salva e faz upload
        image_bytes = image_to_bytes(image)
        url = upload_to_minio(image_bytes, "image.png")
        
        return {
            "status": "success",
            "url": url,
            "model": request.model,
            "seed": request.seed,
            "parameters": {
                "width": request.width,
                "height": request.height,
                "steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na geração: {str(e)}"
        )

@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """Lista modelos disponíveis."""
    try:
        return {
            "models": [
                {
                    "id": "stabilityai/stable-diffusion-xl-base-1.0",
                    "name": "SDXL 1.0",
                    "type": "base"
                },
                {
                    "id": "stabilityai/stable-diffusion-xl-refiner-1.0",
                    "name": "SDXL 1.0 Refiner",
                    "type": "refiner"
                }
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao listar modelos: {str(e)}"
        )

@router.get("/controlnet/models")
async def list_controlnet_models() -> Dict[str, Any]:
    """Lista modelos ControlNet disponíveis."""
    try:
        return {
            "models": [
                {
                    "id": "lllyasviel/control_v11p_sd15_canny",
                    "name": "Canny Edge",
                    "type": "controlnet"
                },
                {
                    "id": "lllyasviel/control_v11p_sd15_openpose",
                    "name": "OpenPose",
                    "type": "controlnet"
                }
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao listar modelos ControlNet: {str(e)}"
        )

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Verifica saúde do serviço."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "gpu_available": torch.cuda.is_available()
    } 