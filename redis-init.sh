#!/bin/bash

# Configurar sistema
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag

# Configurar sysctl
sysctl -w net.core.somaxconn=65535
sysctl -w vm.overcommit_memory=1

# Aplicar configurações
sysctl -p

# Verificar status
echo "Configurações aplicadas:"
echo "vm.overcommit_memory = $(cat /proc/sys/vm/overcommit_memory)"
echo "transparent_hugepage = $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
echo "net.core.somaxconn = $(cat /proc/sys/net/core/somaxconn)" 