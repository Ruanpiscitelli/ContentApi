#!/usr/bin/env python3
"""
Script para download dos modelos do Fish Speech.
Requer huggingface-cli instalado.
"""
import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = Path("checkpoints/fish-speech-1.5")
MODEL_REPO = "fishaudio/fish-speech-1.5"

def check_huggingface_cli():
    """Verifica se huggingface-cli está instalado."""
    try:
        subprocess.run(["huggingface-cli", "--version"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def download_models():
    """Download dos modelos do Fish Speech."""
    if not check_huggingface_cli():
        logger.error("huggingface-cli não está instalado. Instalando...")
        try:
            subprocess.run(["pip", "install", "huggingface-hub"], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro ao instalar huggingface-hub: {e}")
            return False

    # Cria diretório se não existir
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Download dos modelos
    try:
        logger.info("Baixando modelos do Fish Speech...")
        subprocess.run(
            [
                "huggingface-cli", "download",
                MODEL_REPO,
                "--local-dir", str(CHECKPOINTS_DIR),
                "--local-dir-use-symlinks", "False"
            ],
            check=True
        )
        logger.info("Download concluído com sucesso!")
        
        # Verifica arquivos necessários
        required_files = [
            "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        missing_files = [
            f for f in required_files 
            if not (CHECKPOINTS_DIR / f).exists()
        ]
        
        if missing_files:
            logger.warning(f"Arquivos faltando: {missing_files}")
            return False
            
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao baixar modelos: {e}")
        return False

if __name__ == "__main__":
    download_models() 