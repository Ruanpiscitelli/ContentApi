"""
Modelos de banco de dados para o dashboard.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    """Modelo de usuário do dashboard."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    api_keys = relationship("APIKey", back_populates="user")

class APIKey(Base):
    """Modelo de chave API."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    name = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    user = relationship("User", back_populates="api_keys")

class LogEntry(Base):
    """Modelo para logs do sistema."""
    __tablename__ = "logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String)
    service = Column(String)
    message = Column(String)
    details = Column(String, nullable=True) 