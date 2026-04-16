"""
Pydantic Schemas
================
Request/response models for all API endpoints.
All models include Arabic field equivalents for bilingual support.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_AR = {
    SeverityLevel.LOW: "منخفض",
    SeverityLevel.MEDIUM: "متوسط",
    SeverityLevel.HIGH: "عالٍ",
    SeverityLevel.CRITICAL: "حرج",
}

class InspectionStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"


STATUS_AR = {
    InspectionStatus.COMPLIANT: "مطابق",
    InspectionStatus.NON_COMPLIANT: "غير مطابق",
    InspectionStatus.NEEDS_REVIEW: "يحتاج مراجعة",
}


# ── Detection Schemas ─────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float = Field(..., ge=0, description="Top-left x coordinate (normalized 0-1)")
    y1: float = Field(..., ge=0, description="Top-left y coordinate (normalized 0-1)")
    x2: float = Field(..., le=1, description="Bottom-right x coordinate (normalized 0-1)")
    y2: float = Field(..., le=1, description="Bottom-right y coordinate (normalized 0-1)")

    @property
    def area(self) -> float:
        return max(0, (self.x2 - self.x1) * (self.y2 - self.y1))


class Detection(BaseModel):
    """Single detected object."""
    detection_id: UUID = Field(default_factory=uuid4)
    class_id: int
    class_name: str
    class_name_ar: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBox
    severity: SeverityLevel
    severity_ar: str
    is_violation: bool
    description: str
    description_ar: str

    model_config = {"populate_by_name": True}


class DetectionRequest(BaseModel):
    confidence_threshold: float = Field(default=0.45, ge=0.1, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.1, le=1.0)
    return_annotated_image: bool = False


class DetectionResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str
    detections: List[Detection]
    total_detections: int
    violation_count: int
    processing_time_ms: float
    image_width: int
    image_height: int
    annotated_image_base64: Optional[str] = None

    @field_validator("total_detections", mode="before")
    @classmethod
    def set_total(cls, v: int, info: Any) -> int:
        return v


# ── Comparison Schemas ────────────────────────────────────────────────────────

class ChangeRegion(BaseModel):
    region_id: int
    bounding_box: BoundingBox
    area_pixels: int
    change_type: str           # e.g., "addition", "removal", "modification"
    change_type_ar: str
    severity: SeverityLevel
    severity_ar: str


class ComparisonResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ssim_score: float = Field(..., ge=0.0, le=1.0,
                              description="Structural similarity (1=identical)")
    change_percentage: float = Field(..., ge=0.0, le=100.0)
    has_significant_changes: bool
    change_regions: List[ChangeRegion]
    before_violations: List[Detection]
    after_violations: List[Detection]
    new_violations: List[Detection]
    resolved_violations: List[Detection]
    processing_time_ms: float
    diff_image_base64: Optional[str] = None


# ── Compliance Report Schemas ─────────────────────────────────────────────────

class ComplianceRule(BaseModel):
    rule_id: str
    rule_name: str
    rule_name_ar: str
    passed: bool
    weight: float = Field(..., ge=0.0, le=1.0)
    details: str
    details_ar: str


class ComplianceResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    location_id: Optional[str] = None
    inspector_id: Optional[str] = None
    overall_score: float = Field(..., ge=0.0, le=100.0)
    status: InspectionStatus
    status_ar: str
    rules_evaluated: List[ComplianceRule]
    detections: List[Detection]
    violation_summary: Dict[str, int]
    recommendations: List[str]
    recommendations_ar: List[str]
    processing_time_ms: float
    model_version: str
    explainability: Dict[str, Any]


# ── Audit Log Schema ──────────────────────────────────────────────────────────

class AuditLogEntry(BaseModel):
    log_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: UUID
    endpoint: str
    user_id: Optional[str] = None
    input_hash: str            # SHA256 of input image (no PII)
    model_version: str
    decision_summary: Dict[str, Any]
    confidence_scores: List[float]
    processing_time_ms: float
    environment: str


# ── Health & Model Info ───────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    model_name: str
    version: str
    framework: str
    num_classes: int
    class_names: List[str]
    input_size: int
    device: str
    loaded_at: datetime
    mlflow_run_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    model_loaded: bool
    uptime_seconds: float
