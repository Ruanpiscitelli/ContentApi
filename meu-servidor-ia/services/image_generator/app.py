# services/image_generator/app.py

import asyncio
import base64
import json
import os
from io import BytesIO

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import Optional, List

# Importa os pipelines do diffusers e torch
from diffusers import StableDiffusionPipeline  # Exemplo para SDXL; para Flux, você pode ter outro pipeline ou configurar de forma diferente
import torch

app = FastAPI()

##############################
# 1. Funções de Autenticação #
##############################

def verify_bearer_token(authorization: str = Header(...)):
    """
    Valida o header Authorization e extrai o token.
    Em produção, substitua "SEU_TOKEN_SEGREDO" por um token seguro ou utilize variáveis de ambiente.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token inválido")
    token = authorization.split(" ")[1]
    if token != "SEU_TOKEN_SEGREDO":
        raise HTTPException(status_code=401, detail="Não autorizado")
    return token

#####################################
# 2. Modelo de Requisição e Templates #
#####################################

class ImageRequest(BaseModel):
    prompt: str
    template_id: Optional[str] = None  
    model: Optional[str] = "sdxl"       # Pode ser "sdxl" ou "flux"
    loras: Optional[List[dict]] = []     # Lista de dicionários com detalhes dos LoRAs, ex: [{"path": "loras/lora1.pt", "scale": 0.8}, {"path": "loras/lora2.pt", "scale": 1.0}]
    num_inference_steps: Optional[int] = 50
    # Outros parâmetros (como guidance_scale) podem ser adicionados conforme necessário

def load_template(template_id: str) -> dict:
    """
    Carrega um template (arquivo JSON) do diretório 'templates/'.
    Exemplo de template:
    {
      "template_id": "image_template1",
      "model": "sdxl",
      "num_inference_steps": 60,
      "loras": [
         {"path": "loras/lora_face.pt", "scale": 0.8},
         {"path": "loras/lora_style.pt", "scale": 1.0}
      ],
      "guidance_scale": 7.5
    }
    """
    template_path = os.path.join("templates", f"{template_id}.json")
    if not os.path.exists(template_path):
        raise ValueError("Template não encontrado.")
    with open(template_path, "r") as f:
        return json.load(f)

#################################
# 3. Carregamento e Otimização  #
#################################

# Variável global para armazenar o pipeline carregado (cache)
pipe = None

def load_pipeline(model_name: str):
    """
    Carrega o pipeline do modelo base, de acordo com o valor de model_name.
    Se model_name for "flux", carrega o pipeline correspondente (exemplo: de um repositório Flux).
    Caso contrário, assume "sdxl" e carrega o pipeline padrão (neste exemplo, usamos stable-diffusion-v1-4 como placeholder).
    
    Também otimiza o uso de GPU: move para cuda e, se houver mais de uma GPU disponível, utiliza DataParallel.
    """
    global pipe
    try:
        if model_name.lower() == "flux":
            # Exemplo: Carrega o pipeline do Flux (substitua pelo repositório real ou configuração específica)
            pipe = StableDiffusionPipeline.from_pretrained(
                "FluxAI/flux-model", revision="fp16", torch_dtype=torch.float16
            )
        else:
            # Padrão: utiliza o pipeline para SDXL (aqui usamos um placeholder com stable-diffusion-v1-4)
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16
            )
        pipe = pipe.to("cuda")
        if torch.cuda.device_count() > 1:
            # Usa DataParallel para distribuir a carga entre múltiplas GPUs
            pipe.unet = torch.nn.DataParallel(pipe.unet)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o pipeline: {str(e)}")
    return pipe

@app.on_event("startup")
def startup_event():
    # Carrega o pipeline padrão (pode ser ajustado se desejar)
    load_pipeline("sdxl")

#######################################
# 4. Injeção de LoRAs (Simulação/Pseudocódigo)
#######################################
def apply_loras(pipeline, loras: List[dict]):
    """
    Aplica os LoRAs ao pipeline.
    Essa função é um placeholder. Na prática, você deve implementar a lógica para carregar
    e mesclar os pesos dos LoRAs no modelo (por exemplo, ajustando as camadas do UNet).
    
    Parâmetro loras: Lista de dicionários, onde cada dicionário contém:
       - "path": caminho para o arquivo de LoRA
       - "scale": fator de influência do LoRA
    """
    for lora in loras:
        lora_path = lora.get("path")
        scale = lora.get("scale", 1.0)
        # Exemplo: pipeline.unet.apply_lora(lora_path, scale)
        # OBS: Essa função "apply_lora" é ilustrativa. A implementação real depende do método escolhido.
        print(f"Aplicando LoRA: {lora_path} com scale {scale}")
    return pipeline

##############################
# 5. Endpoint de Geração de Imagens
##############################

@app.post("/generate-image")
async def generate_image(request: ImageRequest, token: str = Depends(verify_bearer_token)):
    global pipe

    # Se um template for informado, carrega-o e sobrescreve os parâmetros da requisição
    if request.template_id:
        try:
            template = load_template(request.template_id)
            request.model = template.get("model", request.model)
            request.num_inference_steps = template.get("num_inference_steps", request.num_inference_steps)
            # Suporta múltiplos LoRAs com detalhes; sobrescreve o campo "loras"
            request.loras = template.get("loras", request.loras)
            # Se houver outros parâmetros (ex.: guidance_scale), podem ser extraídos aqui.
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao carregar template: {str(e)}")

    # Se o pipeline não estiver carregado ou se o modelo solicitado for diferente do atual, recarrega-o
    if pipe is None or request.model.lower() not in ["sdxl", "flux"]:
        load_pipeline(request.model)
    elif pipe is None or request.model.lower() != "sdxl":  # Simplesmente ilustrativo: se o modelo mudou, recarrega
        load_pipeline(request.model)

    try:
        # Se houver LoRAs configurados, aplica-os ao pipeline
        if request.loras:
            apply_loras(pipe, request.loras)

        # Executa a inferência de forma assíncrona para maximizar a utilização da GPU
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: pipe(request.prompt, num_inference_steps=request.num_inference_steps)
        )
        image = result.images[0]
        
        # Converte a imagem gerada (objeto PIL) para base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "status": "sucesso",
            "message": "Imagem gerada com sucesso.",
            "image_base64": img_str
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geração da imagem: {str(e)}")