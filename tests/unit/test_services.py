"""
Unit Tests — Detection & Compliance Services
============================================
"""

import numpy as np
import pytest

from app.models.schemas import BoundingBox, Detection, SeverityLevel, SEVERITY_AR
from app.services.compliance_service import ComplianceService
from app.services.detection_service import decode_image, image_sha256


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_detection(
    class_id: int = 0,
    class_name: str = "Illegal Dumping",
    severity: SeverityLevel = SeverityLevel.HIGH,
    is_violation: bool = True,
    confidence: float = 0.85,
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        class_name_ar="انتهاك",
        confidence=confidence,
        bounding_box=BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5),
        severity=severity,
        severity_ar=SEVERITY_AR[severity],
        is_violation=is_violation,
        description="Test violation",
        description_ar="انتهاك تجريبي",
    )


# ── BoundingBox Tests ─────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_area_calculation(self):
        bb = BoundingBox(x1=0.0, y1=0.0, x2=0.5, y2=0.4)
        assert abs(bb.area - 0.2) < 1e-9

    def test_zero_area(self):
        bb = BoundingBox(x1=0.3, y1=0.3, x2=0.3, y2=0.3)
        assert bb.area == 0.0

    def test_full_image(self):
        bb = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        assert abs(bb.area - 1.0) < 1e-9


# ── Image Utilities Tests ─────────────────────────────────────────────────────

class TestImageUtils:
    def _make_jpeg_bytes(self) -> bytes:
        """Generate a minimal valid JPEG in memory."""
        import cv2
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    def test_decode_image_valid(self):
        raw = self._make_jpeg_bytes()
        img, w, h = decode_image(raw)
        assert img is not None
        assert w == 64 and h == 64

    def test_decode_image_invalid(self):
        with pytest.raises(ValueError, match="Could not decode"):
            decode_image(b"not-an-image")

    def test_sha256_deterministic(self):
        data = b"test-image-bytes"
        assert image_sha256(data) == image_sha256(data)

    def test_sha256_different_inputs(self):
        assert image_sha256(b"abc") != image_sha256(b"xyz")


# ── Compliance Service Tests ──────────────────────────────────────────────────

class TestComplianceService:
    def setup_method(self):
        self.svc = ComplianceService()

    def test_perfect_score_no_violations(self):
        result = self.svc.evaluate(
            detections=[],
            model_version="test-v1",
        )
        assert result.overall_score == 100.0
        assert result.status.value == "compliant"

    def test_critical_violation_lowers_score(self):
        det = make_detection(class_id=1, severity=SeverityLevel.CRITICAL, is_violation=True)
        result = self.svc.evaluate([det], model_version="test-v1")
        assert result.overall_score < 100.0
        assert result.status.value in ("non_compliant", "needs_review")

    def test_non_violation_detection_no_penalty(self):
        det = make_detection(class_id=4, is_violation=False, severity=SeverityLevel.LOW)
        result = self.svc.evaluate([det], model_version="test-v1")
        assert result.overall_score == 100.0

    def test_violation_summary_counts(self):
        dets = [
            make_detection(class_id=0, class_name="Illegal Dumping"),
            make_detection(class_id=0, class_name="Illegal Dumping"),
            make_detection(class_id=1, class_name="Blocked Exit"),
        ]
        result = self.svc.evaluate(dets, model_version="test-v1")
        assert result.violation_summary["Illegal Dumping"] == 2
        assert result.violation_summary["Blocked Exit"] == 1

    def test_explainability_fields_present(self):
        result = self.svc.evaluate([], model_version="test-v1")
        exp = result.explainability
        assert "base_score" in exp
        assert "final_score" in exp
        assert "rule_contributions" in exp

    def test_arabic_fields_populated(self):
        det = make_detection(class_id=1, severity=SeverityLevel.CRITICAL)
        result = self.svc.evaluate([det], model_version="test-v1")
        assert result.status_ar in ("غير مطابق", "يحتاج مراجعة", "مطابق")
        assert len(result.recommendations_ar) > 0

    def test_score_clamped_between_0_and_100(self):
        dets = [make_detection(class_id=i % 4, severity=SeverityLevel.CRITICAL) for i in range(10)]
        result = self.svc.evaluate(dets, model_version="test-v1")
        assert 0.0 <= result.overall_score <= 100.0

    def test_multiple_rules_weighted_correctly(self):
        # Only one low-weight rule fails → score should still be decent
        det = make_detection(class_id=2, severity=SeverityLevel.MEDIUM)  # Missing sign (weight=0.20)
        result = self.svc.evaluate([det], model_version="test-v1")
        assert result.overall_score >= 60.0
