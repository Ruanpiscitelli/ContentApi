"""
Este módulo contém os modelos de dados utilizados pelo dashboard.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    """
    Modelo para usuários do dashboard.
    """
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamentos
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    logs = relationship("LogEntry", back_populates="user")

class APIKey(Base):
    """
    Modelo para chaves de API.
    """
    
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String(32), unique=True, index=True, nullable=False)
    description = Column(String(200))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime)
    expires_at = Column(DateTime)
    
    # Relacionamentos
    user = relationship("User", back_populates="api_keys")
    usage_logs = relationship("APIKeyUsage", back_populates="api_key", cascade="all, delete-orphan")

class APIKeyUsage(Base):
    """
    Modelo para registro de uso das chaves de API.
    """
    
    __tablename__ = "api_key_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False)
    endpoint = Column(String(200), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time = Column(Integer)  # em milissegundos
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relacionamentos
    api_key = relationship("APIKey", back_populates="usage_logs")

class LogEntry(Base):
    """
    Modelo para logs do sistema.
    """
    
    __tablename__ = "logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    level = Column(String(10), nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    service = Column(String(50))  # Nome do serviço que gerou o log
    message = Column(String(500), nullable=False)
    details = Column(String(1000))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relacionamentos
    user = relationship("User", back_populates="logs")

class SystemMetric(Base):
    """
    Modelo para métricas do sistema.
    """
    
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    cpu_percent = Column(Integer)
    memory_used = Column(Integer)
    memory_total = Column(Integer)
    disk_used = Column(Integer)
    disk_total = Column(Integer)
    active_connections = Column(Integer)
    requests_per_minute = Column(Integer)
    average_response_time = Column(Integer)  # em milissegundos
    timestamp = Column(DateTime, default=datetime.utcnow)

class Setting(Base):
    """
    Modelo para configurações do sistema.
    """
    
    __tablename__ = "settings"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, index=True, nullable=False)
    value = Column(String(500))
    description = Column(String(200))
    is_editable = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @classmethod
    async def get_value(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Obtém o valor de uma configuração.
        
        Args:
            key (str): Chave da configuração.
            default (Optional[str]): Valor padrão caso a configuração não exista.
        
        Returns:
            Optional[str]: Valor da configuração ou o valor padrão.
        """
        setting = await cls.query.filter(cls.key == key).first()
        return setting.value if setting else default
    
    @classmethod
    async def set_value(cls, key: str, value: str) -> None:
        """
        Define o valor de uma configuração.
        
        Args:
            key (str): Chave da configuração.
            value (str): Valor da configuração.
        """
        setting = await cls.query.filter(cls.key == key).first()
        if setting:
            setting.value = value
        else:
            setting = cls(key=key, value=value)
            cls.session.add(setting)
        await cls.session.commit() 