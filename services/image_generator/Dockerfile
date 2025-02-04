# Atualiza o pip e instala primeiro o PyTorch
RUN pip install --upgrade pip && \
    pip install torch==2.1.0 torchvision==0.16.0

# Instala o restante das dependências
RUN pip install --no-cache-dir -r requirements.txt 