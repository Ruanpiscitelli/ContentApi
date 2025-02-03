#!/usr/bin/env python3
"""
Utilitário para redimensionar vídeos para as dimensões suportadas pelo FastHunyuan.
"""
import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resize_video(input_path: str, output_path: str, target_width: int = 1024, target_height: int = 576):
    """
    Redimensiona um vídeo para as dimensões especificadas.
    
    Args:
        input_path: Caminho do vídeo de entrada
        output_path: Caminho para salvar o vídeo redimensionado
        target_width: Largura desejada (deve ser múltiplo de 64)
        target_height: Altura desejada (deve ser múltiplo de 64)
    """
    try:
        # Valida dimensões
        if target_width % 64 != 0 or target_height % 64 != 0:
            raise ValueError("Dimensões devem ser múltiplos de 64")
            
        # Abre o vídeo
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {input_path}")
            
        # Obtém propriedades do vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Configura o writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path, 
            fourcc, 
            fps, 
            (target_width, target_height)
        )
        
        # Processa cada frame
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Redimensiona o frame
            resized = cv2.resize(frame, (target_width, target_height))
            out.write(resized)
            
            frame_count += 1
            if frame_count % 10 == 0:
                logger.info(f"Processado: {frame_count}/{total_frames} frames")
                
        # Limpa
        cap.release()
        out.release()
        
        logger.info(f"Vídeo redimensionado salvo em: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao redimensionar vídeo: {e}")
        return False
    
def process_directory(input_dir: str, output_dir: str, width: int = 1024, height: int = 576):
    """
    Processa todos os vídeos em um diretório.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    for video_file in input_path.glob('*'):
        if video_file.suffix.lower() in video_extensions:
            output_file = output_path / f"resized_{video_file.name}"
            logger.info(f"Processando: {video_file}")
            resize_video(str(video_file), str(output_file), width, height)

def main():
    parser = argparse.ArgumentParser(description='Redimensiona vídeos para uso com FastHunyuan')
    parser.add_argument('--input', required=True, help='Arquivo ou diretório de entrada')
    parser.add_argument('--output', required=True, help='Arquivo ou diretório de saída')
    parser.add_argument('--width', type=int, default=1024, help='Largura desejada (múltiplo de 64)')
    parser.add_argument('--height', type=int, default=576, help='Altura desejada (múltiplo de 64)')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        process_directory(args.input, args.output, args.width, args.height)
    else:
        resize_video(args.input, args.output, args.width, args.height)

if __name__ == "__main__":
    main() 