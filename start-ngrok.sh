#!/bin/bash

# Verificar se está rodando como root
if [ "$EUID" -ne 0 ]; then 
  echo "Por favor, execute como root (sudo ./start-ngrok.sh)"
  exit 1
fi

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Criar diretório de logs
mkdir -p /var/log/ngrok

# Criar arquivo de serviço systemd
cat > /etc/systemd/system/ngrok.service << EOL
[Unit]
Description=ngrok tunnel service
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/ngrok http 8006 \
    --log=/var/log/ngrok/ngrok.log \
    --log-format=json \
    --log-level=info
Restart=always
RestartSec=10
User=root
WorkingDirectory=/root

[Install]
WantedBy=multi-user.target
EOL

# Ajustar permissões
chmod 644 /etc/systemd/system/ngrok.service

# Recarregar configurações do systemd
systemctl daemon-reload

# Habilitar e iniciar serviço
systemctl enable ngrok
systemctl start ngrok

# Aguardar ngrok iniciar
echo "Aguardando ngrok iniciar..."
sleep 10

# Mostrar status
echo "Status do serviço ngrok:"
systemctl status ngrok

# Mostrar URL pública
echo -e "\nURLs do ngrok:"
curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*' | cut -d'"' -f4

echo -e "\nComandos úteis:"
echo "- Ver logs: tail -f /var/log/ngrok/ngrok.log"
echo "- Parar serviço: systemctl stop ngrok"
echo "- Reiniciar serviço: systemctl restart ngrok"
echo "- Ver status: systemctl status ngrok" 