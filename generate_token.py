#!/usr/bin/env python3

import jwt
from datetime import datetime, timedelta

# Criar payload do token
payload = {
    "sub": "admin",
    "exp": datetime.utcnow() + timedelta(days=1)
}

# Configurações
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"

# Gerar token
token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# Imprimir o token formatado
print(f"\nBearer {token}\n") 