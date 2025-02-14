"""
Utilitários para processamento de texto no serviço de voz
"""
import re
from typing import List
import logging

logger = logging.getLogger(__name__)

def split_text_smart(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Divide texto em chunks de forma inteligente, respeitando pontuação e estrutura.
    
    Args:
        text: Texto para dividir
        max_chunk_size: Tamanho máximo de cada chunk
        
    Returns:
        List[str]: Lista de chunks de texto
    """
    # Remove espaços extras
    text = ' '.join(text.split())
    
    # Se o texto é pequeno, retorna como único chunk
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    # Padrão para encontrar sentenças
    sentence_pattern = r'[^.!?]+[.!?]+'
    sentences = re.findall(sentence_pattern, text)
    
    # Se não encontrou sentenças, usa vírgulas
    if not sentences:
        sentences = text.split(',')
        # Se ainda não tem divisões, divide por espaços
        if len(sentences) == 1:
            sentences = text.split()
    
    for sentence in sentences:
        sentence = sentence.strip()
        sentence_size = len(sentence)
        
        # Se a sentença é maior que o tamanho máximo, divide por espaços
        if sentence_size > max_chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Divide sentença grande em palavras
            words = sentence.split()
            temp_chunk = []
            temp_size = 0
            
            for word in words:
                word_size = len(word) + 1  # +1 para o espaço
                if temp_size + word_size > max_chunk_size:
                    if temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                    temp_chunk = [word]
                    temp_size = word_size
                else:
                    temp_chunk.append(word)
                    temp_size += word_size
            
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
            
        # Se adicionar a sentença excede o tamanho máximo
        elif current_size + sentence_size + 1 > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
            
        # Adiciona a sentença ao chunk atual
        else:
            current_chunk.append(sentence)
            current_size += sentence_size + 1
    
    # Adiciona o último chunk se existir
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Validação final
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    logger.debug(
        f"Texto dividido em {len(chunks)} chunks "
        f"(média de {sum(len(c) for c in chunks)/len(chunks):.0f} caracteres/chunk)"
    )
    
    return chunks

def preprocess_text(text: str) -> str:
    """
    Pré-processa texto para síntese de voz.
    
    Args:
        text: Texto para processar
        
    Returns:
        str: Texto processado
    """
    # Remove espaços extras
    text = ' '.join(text.split())
    
    # Normaliza pontuação
    text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove caracteres inválidos
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Normaliza números
    text = re.sub(r'(\d+)', lambda m: str(int(m.group(1))), text)
    
    return text.strip()

def get_text_stats(text: str) -> dict:
    """
    Retorna estatísticas sobre o texto.
    
    Args:
        text: Texto para analisar
        
    Returns:
        dict: Estatísticas do texto
    """
    # Contagem básica
    chars = len(text)
    words = len(text.split())
    sentences = len(re.findall(r'[.!?]+', text))
    
    # Estimativa de tempo
    words_per_second = 2.5  # Média de palavras por segundo na fala
    estimated_duration = words / words_per_second
    
    return {
        "caracteres": chars,
        "palavras": words,
        "sentencas": sentences,
        "duracao_estimada": estimated_duration
    } 