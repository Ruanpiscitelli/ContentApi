from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./data.db")
    # Restante das configurações... 