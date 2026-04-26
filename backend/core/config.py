"""Application configuration."""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Medical Statistical Analysis Platform"
    DEBUG: bool = True

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # File Upload
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".xlsx", ".xls"]
    UPLOAD_DIR: str = "uploads"

    # LLM Settings
    ANTHROPIC_API_KEY: str = ""
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    LLM_MAX_TOKENS: int = 4096

    # Statistical Settings
    SIGNIFICANCE_LEVEL: float = 0.05
    CONFIDENCE_LEVEL: float = 0.95

    # Output Settings
    OUTPUT_DIR: str = "outputs"
    FIGURES_DIR: str = "outputs/figures"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


settings = Settings()

# Create directories if they don't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.FIGURES_DIR, exist_ok=True)
