#!/usr/bin/env python3
"""
Script para download dos modelos necessários do Fish Speech.
Requer huggingface-cli instalado.
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurações
MODEL_REPO = "fishaudio/fish-speech-1.5"
MODEL_DIR = Path("models/fish-speech-1.5")

def check_huggingface_cli():
    """Verifica se huggingface-cli está instalado."""
    try:
        subprocess.run(["huggingface-cli", "--version"], 
                      check=True, 
                      capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_huggingface_cli():
    """Instala huggingface-cli via pip."""
    try:
        logger.info("Instalando huggingface-hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub"],
                      check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao instalar huggingface-hub: {e}")
        return False

def download_models():
    """Download modelos do Hugging Face."""
    logger.info("Iniciando download dos modelos do Fish Speech...")
    
    # Verifica e instala huggingface-cli se necessário
    if not check_huggingface_cli():
        if not install_huggingface_cli():
            raise RuntimeError("Não foi possível instalar huggingface-cli")
    
    # Criar diretórios necessários
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Diretório do modelo: {MODEL_DIR.absolute()}")
    
    cmd = [
        "huggingface-cli", "download",
        "--resume-download",
        MODEL_REPO,
        "--local-dir", str(MODEL_DIR.absolute())
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("Download concluído com sucesso!")
        
        # Verificar se os arquivos foram baixados
        files = list(MODEL_DIR.glob("*"))
        if not files:
            raise Exception("Nenhum arquivo foi baixado")
            
        logger.info(f"Arquivos baixados: {[f.name for f in files]}")
        
        # Verifica arquivos essenciais
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"  # Arquivo de pesos do modelo
        ]
        
        missing = [f for f in required_files if not (MODEL_DIR / f).exists()]
        if missing:
            raise Exception(
                f"Arquivos essenciais faltando: {missing}\n"
                "ATENÇÃO: O arquivo de pesos do modelo é necessário para o funcionamento correto do gerador de voz."
            )
            
    except Exception as e:
        logger.error(f"Erro ao baixar modelos: {e}")
        raise

if __name__ == "__main__":
    try:
        download_models()
    except Exception as e:
        logger.error(f"Falha ao baixar modelos: {e}")
        sys.exit(1) 