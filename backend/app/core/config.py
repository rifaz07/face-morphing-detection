from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    PROJECT_NAME: str = "Face Morphing Detection API"
    API_V1_PREFIX: str = "/api/v1"
    VERSION: str = "0.1.0"
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str = "postgresql://fmd_user:password@fmd-postgres:5432/face_morphing_db"

    # CORS — accepts a comma-separated string or a JSON list
    CORS_ORIGINS: str = "http://localhost:3000"

    # ------------------------------------------------------------------
    # ML Pipeline — Image Validation (Module 1)
    # ------------------------------------------------------------------
    MAX_IMAGE_SIZE_MB: int = 10
    MIN_IMAGE_DIMENSION: int = 100
    MAX_IMAGE_DIMENSION: int = 4096
    ALLOWED_IMAGE_FORMATS: List[str] = ["JPEG", "PNG", "WEBP"]

    # ------------------------------------------------------------------
    # ML Pipeline — Face Detection / Haar Cascade (Module 2)
    # ------------------------------------------------------------------
    HAAR_SCALE_FACTOR: float = 1.1
    HAAR_MIN_NEIGHBORS: int = 5
    HAAR_MIN_FACE_SIZE: int = 30

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, value: str) -> str:
        return value

    @property
    def cors_origins_list(self) -> List[str]:
        """Return CORS origins as a list, splitting on commas."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.ENV.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
