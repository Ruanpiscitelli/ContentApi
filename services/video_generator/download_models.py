#!/usr/bin/env python3
"""
Script para download dos modelos do FastHunyuan.
"""
import os
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = Path("checkpoints/fasthunyuan")
MODEL_REPO = "FastVideo/FastHunyuan-diffusers"

def download_models():
    """Download dos modelos do FastHunyuan."""
    try:
        logger.info("Baixando modelos do FastHunyuan...")
        
        # Cria diretório se não existir
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download usando huggingface_hub
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=str(CHECKPOINTS_DIR),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        logger.info("Download concluído com sucesso!")
        
        # Verifica arquivos necessários
        required_files = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "tokenizer/tokenizer_config.json",
            "unet/config.json",
            "vae/config.json"
        ]
        
        missing_files = [
            f for f in required_files 
            if not (CHECKPOINTS_DIR / f).exists()
        ]
        
        if missing_files:
            logger.warning(f"Arquivos faltando: {missing_files}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Erro ao baixar modelos: {e}")
        return False

if __name__ == "__main__":
    download_models() 