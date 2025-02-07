# Fish Speech Models

Este diretório armazena os modelos do Fish Speech necessários para a geração de voz.

## Download dos Modelos

Para baixar os modelos, execute o script `download_models.py`:

```bash
python download_models.py
```

### Pré-requisitos

1. Git LFS instalado:
   - MacOS: `brew install git-lfs`
   - Ubuntu: `apt-get install git-lfs`
   - Windows: https://git-lfs.com

2. Espaço em disco:
   - Aproximadamente 5GB para os modelos completos

### Estrutura de Arquivos

Após o download, você terá:

```
models/fish-speech-1.4/
  ├── config.json           # Configuração do modelo
  ├── pytorch_model.bin     # Pesos do modelo
  ├── tokenizer.json        # Configuração do tokenizer
  └── special_tokens_map.json
```

### Fonte dos Modelos

Os modelos são baixados do repositório oficial do Fish Speech no Hugging Face:
https://huggingface.co/fishaudio/fish-speech-1.4

### Notas

- Os arquivos de modelo não são versionados no Git devido ao seu tamanho
- Execute o script de download antes de iniciar o serviço
- Verifique a documentação oficial para mais detalhes sobre os modelos 