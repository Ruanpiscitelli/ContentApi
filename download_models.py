#!/usr/bin/env python3
"""
Script para baixar os modelos e os arquivos adicionais (assets) necessários,
além de instalar as dependências do projeto.

Este script realiza as tarefas a seguir:
1. Cria uma estrutura de diretórios para armazenar os modelos, organizados por categoria:
   - imagem: Modelos para geração de imagens (ex.: SDXL).
   - texto: Modelos para geração de texto.
   - video: Modelos para geração de vídeo.
   - audio: Modelos para geração de voz.
2. Para cada modelo, clona o repositório correspondente do Hugging Face utilizando Git LFS.
   Se o repositório já tiver sido clonado (o diretório existir),
   ele pula a etapa para evitar downloads repetidos.
3. Para os modelos que necessitam de arquivos extras (por exemplo, VAE para SDXL ou Tokenizer para Fish Speech),
   o script baixa os arquivos adicionais se eles não existirem.
4. Instala as dependências do projeto a partir do arquivo requirements.txt (se disponível).

Requisitos:
- Git com suporte ao Git LFS instalado.
- Python 3.8+.
- A biblioteca 'requests' para download dos arquivos extras.
"""

import os
import subprocess
import sys
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Importa a biblioteca requests para download de assets adicionais.
try:
    import requests
except ImportError:
    print("A biblioteca 'requests' não está instalada. Instalando agora...")
    subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
    import requests


def install_dependencies():
    """
    Instala as dependências do projeto a partir do arquivo requirements.txt,
    caso este exista na raiz do projeto.
    """
    requirements_path = os.path.join(os.getcwd(), "requirements.txt")
    if os.path.exists(requirements_path):
        print("Instalando dependências do projeto a partir do requirements.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path], check=True)
    else:
        print("Arquivo requirements.txt não encontrado. A instalação de dependências será ignorada.")


def clone_repo(repo_identifier, target_dir):
    """
    Clona o repositório do Hugging Face utilizando Git LFS.

    Args:
        repo_identifier (str): Identificador do repositório no Hugging Face,
            por exemplo, 'cagliostrolab/animagine-xl-4.0'.
        target_dir (str): Caminho do diretório onde o repositório será clonado.
    """
    repo_url = f"https://huggingface.co/{repo_identifier}"
    print(f"Clonando o repositório {repo_url} em {target_dir}...")
    # Se o diretório já existir, a clonagem será ignorada.
    if os.path.exists(target_dir):
        print(f"O diretório {target_dir} já existe, pulando clonagem.")
    else:
        subprocess.run(["git", "lfs", "clone", repo_url, target_dir], check=True)
    # Após a clonagem, baixa os assets adicionais, se houver.
    download_additional_assets(repo_identifier, target_dir)


def create_session_with_retry():
    """Cria uma sessão HTTP com retry automático."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session


def download_file(url, target_file):
    """
    Faz o download do arquivo com retry automático.
    """
    print(f"Baixando arquivo de {url} para {target_file} ...")
    session = create_session_with_retry()
    
    try:
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(target_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Falha ao baixar {url}. Erro: {str(e)}")


def download_additional_assets(repo_identifier, target_dir):
    """
    Verifica e baixa assets adicionais com URLs otimizadas.
    """
    additional_assets = {
        "John6666/ultimate-realistic-mix-v2-sdxl": [
            # VAE otimizado do madebyollin
            ("vae.safetensors", "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors"),
            # VAE original como fallback
            ("vae.original.safetensors", "https://huggingface.co/John6666/ultimate-realistic-mix-v2-sdxl/resolve/main/vae.safetensors")
        ],
        "fishaudio/fish-speech-1.5": [
            ("tokenizer.json", "https://huggingface.co/fishaudio/fish-speech-1.5/resolve/main/tokenizer.json")
        ]
    }
    if repo_identifier in additional_assets:
        assets = additional_assets[repo_identifier]
        for asset_filename, asset_url in assets:
            target_file = os.path.join(target_dir, asset_filename)
            if not os.path.exists(target_file):
                download_file(asset_url, target_file)
                print(f"Asset '{asset_filename}' baixado para {target_dir}.")
            else:
                print(f"Asset '{asset_filename}' já existe em {target_dir}, pulando o download.")


def main():
    """
    Função principal que define os modelos a serem baixados e organiza a
    estrutura de diretórios.

    As categorias definidas são:
        - imagem: Modelos para geração de imagens SDXL.
        - texto: Modelos para geração de texto.
        - video: Modelo para geração de vídeo.
        - audio: Modelo para geração de voz.
    """
    # Dicionário com as categorias e seus respectivos repositórios.
    modelos = {
        "imagem": [
            "cagliostrolab/animagine-xl-4.0",
            "John6666/ultimate-realistic-mix-v2-sdxl"
        ],
        "texto": [
            "openbmb/MiniCPM-o-2_6",
            "huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated",
            "marketeam/LLa-Marketing",
            "NobodySpecial/Qwen2.5-72B-exl2-5.0bpw",
            "FPHam/StoryCrafter",
            "meta-llama/Llama-3.2-3B-Instruct"
        ],
        "video": [
            "FastVideo/FastHunyuan"
        ],
        "audio": [
            "fishaudio/fish-speech-1.5"
        ]
    }

    # Cria o diretório base "models" na raiz do projeto.
    base_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(base_dir, exist_ok=True)

    # Itera em cada categoria e clona os respectivos repositórios.
    for categoria, repos in modelos.items():
        categoria_dir = os.path.join(base_dir, categoria)
        os.makedirs(categoria_dir, exist_ok=True)
        for repo in repos:
            # O nome do diretório destino é o nome do repositório (última parte do identificador).
            repo_name = repo.split("/")[-1]
            target_dir = os.path.join(categoria_dir, repo_name)
            clone_repo(repo, target_dir)

    print("Download dos modelos e assets concluído.")

    # Instala as dependências do projeto, se houver o arquivo requirements.txt.
    install_dependencies()


if __name__ == "__main__":
    main()