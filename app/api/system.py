"""
Models & Audit API Router
=========================
Endpoints for model metadata, health, and audit log retrieval.
"""

from fastapi import APIRouter, Depends, Query

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import HealthResponse, ModelInfo
from app.services.audit_service import AuditService, get_audit_service
from app.services.detection_service import DetectionService, get_detection_service
import time
import datetime

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(tags=["System"])

_start_time = time.time()


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health(
    detector: DetectionService = Depends(get_detection_service),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        model_loaded=detector.is_loaded(),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/models/info", response_model=ModelInfo, summary="Active model metadata")
async def model_info(
    detector: DetectionService = Depends(get_detection_service),
) -> ModelInfo:
    if not detector.is_loaded():
        detector.load_model()
    info = detector.get_model_info()
    return ModelInfo(
        model_name="SmartInspection-YOLOv8",
        version=info["version"],
        framework="PyTorch / Ultralytics",
        num_classes=info["num_classes"],
        class_names=info["class_names"],
        input_size=settings.IMAGE_SIZE,
        device=info["device"],
        loaded_at=info["loaded_at"] or datetime.datetime.utcnow(),
    )


@router.get("/audit/logs", summary="Retrieve recent AI decision audit logs")
async def audit_logs(
    limit: int = Query(default=20, ge=1, le=200),
    audit: AuditService = Depends(get_audit_service),
) -> list:
    """
    Returns the most recent audit log entries.
    Each entry contains input hash, model version, decision summary,
    and confidence distribution — no PII stored.
    """
    return audit.get_recent(limit=limit)
