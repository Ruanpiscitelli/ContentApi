"""
Security utilities shared across all services.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class SecurityConfig(BaseModel):
    """Security configuration settings."""
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
class TokenManager:
    """Manages JWT token operations."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.security = HTTPBearer()
        
    def create_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Creates a new JWT token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES)
            
        to_encode.update({"exp": expire})
        
        try:
            encoded_jwt = jwt.encode(
                to_encode, 
                self.config.SECRET_KEY, 
                algorithm=self.config.ALGORITHM
            )
            return encoded_jwt
        except Exception as e:
            logger.error(f"Error creating token: {e}")
            raise HTTPException(
                status_code=500,
                detail="Could not create access token"
            )
            
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verifies and decodes a JWT token."""
        try:
            payload = jwt.decode(
                token, 
                self.config.SECRET_KEY, 
                algorithms=[self.config.ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials"
            )
            
    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
    ) -> Dict[str, Any]:
        """FastAPI dependency for getting current authenticated user."""
        try:
            token = credentials.credentials
            payload = self.verify_token(token)
            return payload
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
            
def create_token_manager(config: SecurityConfig) -> TokenManager:
    """Creates a TokenManager instance with the given config."""
    return TokenManager(config) 