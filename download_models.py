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

Nota: exllama foi removido dos requirements devido a problemas de compatibilidade.
Se necessário, pode ser instalado manualmente ou substituído por alternativas como:
- exllamav2
- llama-cpp-python
- transformers com quantização nativa
"""

import os
import subprocess
import sys
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

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
    Clona o repositório do Hugging Face usando git clone com LFS.

    Args:
        repo_identifier (str): Identificador do repositório no Hugging Face
        target_dir (str): Caminho do diretório onde o repositório será clonado
    """
    # Verifica se devemos pular este repositório
    if "Qwen2.5-72B" in repo_identifier:
        print(f"Pulando download do modelo {repo_identifier} (muito grande)")
        return

    repo_url = f"https://huggingface.co/{repo_identifier}"
    print(f"Clonando o repositório {repo_url} em {target_dir}...")
    
    if os.path.exists(target_dir):
        print(f"O diretório {target_dir} já existe, pulando download...")
        return
    else:
        try:
            # Configura LFS antes do clone
            subprocess.run(["git", "lfs", "install"], check=True)
            
            # Clone com retry em caso de timeout
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    subprocess.run([
                        "git", "clone",
                        "--recurse-submodules",
                        "--depth", "1",
                        repo_url, target_dir
                    ], check=True, timeout=600)  # 10 minutos timeout
                    
                    # Pull LFS files
                    subprocess.run([
                        "git", "-C", target_dir,
                        "lfs", "pull"
                    ], check=True, timeout=600)
                    
                    break
                except subprocess.TimeoutExpired:
                    print(f"Timeout no attempt {attempt + 1}/{max_retries}, tentando novamente...")
                    if os.path.exists(target_dir):
                        import shutil
                        shutil.rmtree(target_dir)
                    if attempt == max_retries - 1:
                        raise
        except Exception as e:
            print(f"Erro ao clonar repositório: {e}")
            raise

    # Após a clonagem, baixa os assets adicionais
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


def download_with_retry(url, target_file, max_retries=3, timeout=300):
    """
    Faz download com retry automático.
    
    Args:
        url: URL do arquivo
        target_file: Caminho onde salvar
        max_retries: Número máximo de tentativas
        timeout: Timeout em segundos
    """
    for attempt in range(max_retries):
        try:
            session = create_session_with_retry()
            response = session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                with open(target_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"Tentativa {attempt + 1}/{max_retries} falhou: {e}")
            if attempt == max_retries - 1:
                raise
            import time
            time.sleep(5 * (attempt + 1))  # Backoff exponencial


def download_additional_assets(repo_identifier, target_dir):
    """Verifica e baixa assets adicionais com retry."""
    # Pula se for o Qwen2.5
    if "Qwen2.5-72B" in repo_identifier:
        return

    additional_assets = {
        "John6666/ultimate-realistic-mix-v2-sdxl": [
            ("vae.safetensors", [
                "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
                "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors"
            ])
        ],
        "fishaudio/fish-speech-1.5": [
            ("tokenizer.json", [
                "https://huggingface.co/fishaudio/fish-speech-1.5/resolve/main/tokenizer.json"
            ])
        ]
    }
    
    if repo_identifier in additional_assets:
        assets = additional_assets[repo_identifier]
        for asset_filename, urls in assets:
            target_file = os.path.join(target_dir, asset_filename)
            if not os.path.exists(target_file):
                success = False
                for url in urls:
                    try:
                        print(f"Tentando baixar {asset_filename} de {url}")
                        download_with_retry(url, target_file)
                        print(f"Asset '{asset_filename}' baixado com sucesso")
                        success = True
                        break
                    except Exception as e:
                        print(f"Erro ao baixar de {url}: {e}")
                if not success:
                    raise Exception(f"Falha ao baixar {asset_filename} de todas as URLs")
            else:
                print(f"Asset '{asset_filename}' já existe em {target_dir}, pulando download...")


def validate_model(target_dir):
    """
    Valida se o modelo foi baixado corretamente.
    
    Args:
        target_dir: Diretório do modelo
    Returns:
        bool: True se válido, False caso contrário
    """
    required_files = ['config.json', 'pytorch_model.bin']
    for file in required_files:
        if not os.path.exists(os.path.join(target_dir, file)):
            return False
    return True


def cleanup_cache():
    """Limpa caches temporários após downloads."""
    import shutil
    cache_dirs = [
        '~/.cache/huggingface',
        '~/.cache/torch',
        '~/.cache/pip'
    ]
    for cache_dir in cache_dirs:
        path = os.path.expanduser(cache_dir)
        if os.path.exists(path):
            shutil.rmtree(path)


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

    # Limpa o cache após o download
    cleanup_cache()


if __name__ == "__main__":
    main()