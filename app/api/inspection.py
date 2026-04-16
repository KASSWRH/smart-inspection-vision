"""
Inspection API Router
=====================
Endpoints for detection, before/after comparison, and compliance reporting.
All endpoints are versioned under /api/v1/inspect/.
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import ComplianceResponse, ComparisonResponse, DetectionResponse
from app.services.audit_service import AuditService, get_audit_service
from app.services.compliance_service import ComplianceService
from app.services.comparison_service import ComparisonService
from app.services.detection_service import (
    DetectionService,
    decode_image,
    get_detection_service,
    image_sha256,
)

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/inspect", tags=["Inspection"])

MAX_BYTES = settings.MAX_REQUEST_SIZE_MB * 1024 * 1024


# ── Dependency Helpers ────────────────────────────────────────────────────────

def _get_comparison_service(
    detector: DetectionService = Depends(get_detection_service),
) -> ComparisonService:
    return ComparisonService(detector)


def _get_compliance_service() -> ComplianceService:
    return ComplianceService()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Run object/violation detection on a single image",
    response_description="List of detected objects with bounding boxes and severity",
)
async def detect(
    image: UploadFile = File(..., description="JPEG or PNG image to inspect"),
    confidence_threshold: float = Form(default=0.45, ge=0.1, le=1.0),
    iou_threshold: float = Form(default=0.45, ge=0.1, le=1.0),
    return_annotated_image: bool = Form(default=False),
    detector: DetectionService = Depends(get_detection_service),
    audit: AuditService = Depends(get_audit_service),
) -> DetectionResponse:
    """
    Detect violations and objects in an uploaded image.

    - Supports JPEG and PNG formats up to 20 MB.
    - Returns bounding boxes, confidence scores, severity levels.
    - Optionally returns annotated image as base64 JPEG.
    - Bilingual labels (EN + AR) included in every detection.
    """
    raw = await image.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {settings.MAX_REQUEST_SIZE_MB} MB limit.",
        )

    try:
        img, _, _ = decode_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    if not detector.is_loaded():
        detector.load_model()

    result = detector.detect(img, confidence_threshold, iou_threshold, return_annotated_image)

    audit.log(
        request_id=result.request_id,
        endpoint="/inspect/detect",
        input_hash=image_sha256(raw),
        model_version=result.model_version,
        decision_summary={
            "total_detections": result.total_detections,
            "violation_count": result.violation_count,
        },
        confidence_scores=[d.confidence for d in result.detections],
        processing_time_ms=result.processing_time_ms,
    )

    return result


@router.post(
    "/compare",
    response_model=ComparisonResponse,
    summary="Before/after image comparison for change detection",
)
async def compare(
    before_image: UploadFile = File(..., description="Before-state inspection image"),
    after_image: UploadFile = File(..., description="After-state inspection image"),
    confidence_threshold: float = Form(default=0.45),
    comparison_svc: ComparisonService = Depends(_get_comparison_service),
    audit: AuditService = Depends(get_audit_service),
) -> ComparisonResponse:
    """
    Compare before and after images to detect structural changes and violation deltas.

    - Computes SSIM score and highlights changed regions.
    - Identifies new violations introduced and violations resolved.
    - Useful for compliance trend tracking over time.
    """
    before_raw = await before_image.read()
    after_raw = await after_image.read()

    try:
        before_img, _, _ = decode_image(before_raw)
        after_img, _, _ = decode_image(after_raw)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    result = comparison_svc.compare(before_img, after_img, confidence_threshold)

    audit.log(
        request_id=result.request_id,
        endpoint="/inspect/compare",
        input_hash=f"{image_sha256(before_raw)}:{image_sha256(after_raw)}",
        model_version="comparison-v1",
        decision_summary={
            "ssim_score": result.ssim_score,
            "change_percentage": result.change_percentage,
            "new_violations": len(result.new_violations),
            "resolved_violations": len(result.resolved_violations),
        },
        confidence_scores=[],
        processing_time_ms=result.processing_time_ms,
    )

    return result


@router.post(
    "/compliance",
    response_model=ComplianceResponse,
    summary="Full compliance report with scoring and recommendations",
)
async def compliance(
    image: UploadFile = File(...),
    location_id: Optional[str] = Form(default=None),
    inspector_id: Optional[str] = Form(default=None),
    confidence_threshold: float = Form(default=0.45),
    detector: DetectionService = Depends(get_detection_service),
    compliance_svc: ComplianceService = Depends(_get_compliance_service),
    audit: AuditService = Depends(get_audit_service),
) -> ComplianceResponse:
    """
    Run full compliance pipeline:
    1. Detect violations via YOLOv8.
    2. Evaluate against compliance rule set.
    3. Compute weighted score with severity penalties.
    4. Return bilingual recommendations and explainability metadata.
    """
    raw = await image.read()
    try:
        img, _, _ = decode_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    if not detector.is_loaded():
        detector.load_model()

    detection_result = detector.detect(img, confidence_threshold)
    compliance_result = compliance_svc.evaluate(
        detections=detection_result.detections,
        model_version=detection_result.model_version,
        location_id=location_id,
        inspector_id=inspector_id,
        processing_time_ms=detection_result.processing_time_ms,
    )

    audit.log(
        request_id=compliance_result.request_id,
        endpoint="/inspect/compliance",
        input_hash=image_sha256(raw),
        model_version=detection_result.model_version,
        decision_summary={
            "score": compliance_result.overall_score,
            "status": compliance_result.status.value,
            "violations": compliance_result.violation_summary,
        },
        confidence_scores=[d.confidence for d in detection_result.detections],
        processing_time_ms=compliance_result.processing_time_ms,
    )

    return compliance_result
