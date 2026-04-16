"""
Core Configuration Module
=========================
Centralized settings management using Pydantic BaseSettings.
All config values are loaded from environment variables or .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ─────────────────────────────────────────────────────────
    APP_NAME: str = "Smart Inspection Vision System"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── API ──────────────────────────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["*"]
    MAX_REQUEST_SIZE_MB: int = 20

    # ── Security ─────────────────────────────────────────────────────────────
    SECRET_KEY: str = Field(default="change-me-in-production", min_length=32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./inspection.db"

    # ── Model Paths ──────────────────────────────────────────────────────────
    MODEL_DIR: Path = Path("models")
    DETECTION_MODEL_PATH: Path = Path("models/detection/best.pt")
    ONNX_MODEL_PATH: Path = Path("models/export/model.onnx")
    TFLITE_MODEL_PATH: Path = Path("models/export/model.tflite")

    # ── Detection Defaults ───────────────────────────────────────────────────
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.45
    DEFAULT_IOU_THRESHOLD: float = 0.45
    MAX_DETECTIONS: int = 100
    IMAGE_SIZE: int = 640  # YOLOv8 input size

    # ── Comparison (SSIM) ────────────────────────────────────────────────────
    SSIM_CHANGE_THRESHOLD: float = 0.85   # Below = significant change detected
    DIFF_MIN_AREA_PX: int = 500           # Ignore tiny pixel differences

    # ── MLflow ───────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "smart-inspection"

    # ── Audit ────────────────────────────────────────────────────────────────
    AUDIT_LOG_ENABLED: bool = True
    AUDIT_LOG_PATH: Path = Path("logs/audit.jsonl")

    @field_validator("DETECTION_MODEL_PATH", "MODEL_DIR", mode="before")
    @classmethod
    def resolve_path(cls, v: str) -> Path:
        return Path(v)

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
