"""
Script para download dos modelos de text generation.
Faz o download dos modelos do HuggingFace Hub e aplica quantização AWQ.
"""
import os
import json
import logging
import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Diretórios
BASE_DIR = Path(__file__).parent.absolute()
CONFIG_FILE = BASE_DIR / "config/models_config.json"
CACHE_DIR = BASE_DIR / "cache"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

def load_config():
    """Carrega configuração dos modelos."""
    with open(CONFIG_FILE) as f:
        return json.load(f)

def download_model(model_id: str, config: dict):
    """
    Faz download do modelo e tokenizer.
    
    Args:
        model_id: ID do modelo no HuggingFace Hub
        config: Configuração do modelo
    """
    logger.info(f"Iniciando download do modelo {model_id}")
    
    try:
        # Criar diretórios
        cache_dir = BASE_DIR / config["cache_dir"]
        checkpoint_dir = BASE_DIR / config["checkpoint_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Download do modelo
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_dir=checkpoint_dir,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.safetensors"]
        )
        
        # Download do tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Download do modelo {model_id} concluído")
        
        # Aplicar quantização se configurada
        if config.get("quantization") == "awq":
            quantize_model(model_id, config)
            
    except Exception as e:
        logger.error(f"Erro ao baixar modelo {model_id}: {e}")
        raise

def quantize_model(model_id: str, config: dict):
    """
    Aplica quantização AWQ no modelo.
    
    Args:
        model_id: ID do modelo no HuggingFace Hub
        config: Configuração do modelo
    """
    logger.info(f"Iniciando quantização AWQ do modelo {model_id}")
    
    try:
        checkpoint_dir = BASE_DIR / config["checkpoint_dir"]
        
        # Carregar modelo em FP16
        model = AutoGPTQForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Configurar quantização
        model.quantize(
            bits=4,
            group_size=128,
            desc_act=True,
            sym=True
        )
        
        # Salvar modelo quantizado
        model.save_pretrained(
            checkpoint_dir / "quantized",
            use_safetensors=True
        )
        
        logger.info(f"Quantização do modelo {model_id} concluída")
        
    except Exception as e:
        logger.error(f"Erro na quantização do modelo {model_id}: {e}")
        raise

def main():
    """Função principal."""
    try:
        # Carregar configuração
        config = load_config()
        
        # Download de cada modelo
        for model_id, model_config in config["models"].items():
            download_model(model_id, model_config)
            
        logger.info("Download de todos os modelos concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro no processo de download: {e}")
        raise

if __name__ == "__main__":
    main() 