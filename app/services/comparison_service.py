"""
Comparison Service
==================
Performs structural similarity (SSIM) based before/after image comparison
to detect changes, new violations, and resolved issues between two inspection images.
"""

import time
from typing import List, Tuple
from uuid import uuid4

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import (
    BoundingBox,
    ChangeRegion,
    ComparisonResponse,
    Detection,
    SEVERITY_AR,
    SeverityLevel,
)
from app.services.detection_service import DetectionService

logger = get_logger(__name__)
settings = get_settings()

CHANGE_TYPE_MAP = {
    "addition": "إضافة",
    "removal": "إزالة",
    "modification": "تعديل",
}


class ComparisonService:
    """
    Compares before and after inspection images to identify:
    - Overall structural similarity (SSIM)
    - Pixel-level change regions
    - New violations introduced
    - Violations that were resolved
    """

    def __init__(self, detection_service: DetectionService) -> None:
        self._detector = detection_service

    def compare(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        confidence_threshold: float = 0.45,
    ) -> ComparisonResponse:
        """
        Run full before/after comparison pipeline.

        Args:
            before_image: BGR numpy array of the before state.
            after_image: BGR numpy array of the after state.
            confidence_threshold: Detection confidence cutoff.

        Returns:
            ComparisonResponse with SSIM, change regions, and violation delta.
        """
        start = time.perf_counter()

        # Resize after to match before dimensions if needed
        if before_image.shape != after_image.shape:
            after_image = cv2.resize(
                after_image,
                (before_image.shape[1], before_image.shape[0]),
            )

        # ── SSIM Computation ──────────────────────────────────────────────
        ssim_score, diff_map = self._compute_ssim(before_image, after_image)
        change_pct = round((1.0 - ssim_score) * 100, 2)
        has_changes = ssim_score < settings.SSIM_CHANGE_THRESHOLD

        # ── Change Region Extraction ──────────────────────────────────────
        change_regions = self._extract_change_regions(diff_map, before_image.shape)

        # ── Run Detection on Both Images ──────────────────────────────────
        before_resp = self._detector.detect(before_image, confidence_threshold)
        after_resp = self._detector.detect(after_image, confidence_threshold)

        # ── Compute Violation Delta ───────────────────────────────────────
        new_violations, resolved = self._compute_violation_delta(
            before_resp.detections, after_resp.detections
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return ComparisonResponse(
            request_id=uuid4(),
            ssim_score=round(ssim_score, 4),
            change_percentage=change_pct,
            has_significant_changes=has_changes,
            change_regions=change_regions,
            before_violations=[d for d in before_resp.detections if d.is_violation],
            after_violations=[d for d in after_resp.detections if d.is_violation],
            new_violations=new_violations,
            resolved_violations=resolved,
            processing_time_ms=round(elapsed_ms, 2),
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _compute_ssim(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute SSIM and return (score, diff_map)."""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, diff = ssim(gray1, gray2, full=True)
        # Convert diff to 0-255 uint8
        diff_uint8 = (diff * 255).astype(np.uint8)
        return float(score), diff_uint8

    def _extract_change_regions(
        self, diff_map: np.ndarray, image_shape: Tuple[int, ...]
    ) -> List[ChangeRegion]:
        """Find contiguous changed regions in the diff map."""
        h, w = image_shape[:2]
        _, thresh = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove noise with morphological ops
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions: List[ChangeRegion] = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < settings.DIFF_MIN_AREA_PX:
                continue

            x, y, rw, rh = cv2.boundingRect(cnt)
            severity = self._area_to_severity(area)
            change_type = "modification"  # Default; could be refined with deeper logic

            regions.append(
                ChangeRegion(
                    region_id=idx,
                    bounding_box=BoundingBox(
                        x1=x / w, y1=y / h,
                        x2=(x + rw) / w, y2=(y + rh) / h,
                    ),
                    area_pixels=int(area),
                    change_type=change_type,
                    change_type_ar=CHANGE_TYPE_MAP.get(change_type, change_type),
                    severity=severity,
                    severity_ar=SEVERITY_AR[severity],
                )
            )

        logger.debug("Change regions found", count=len(regions))
        return regions

    def _area_to_severity(self, area_px: float) -> SeverityLevel:
        """Heuristic: map changed area size → severity."""
        if area_px > 50_000:
            return SeverityLevel.CRITICAL
        if area_px > 20_000:
            return SeverityLevel.HIGH
        if area_px > 5_000:
            return SeverityLevel.MEDIUM
        return SeverityLevel.LOW

    def _compute_violation_delta(
        self,
        before: List[Detection],
        after: List[Detection],
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        Compare violation class sets to determine:
        - new_violations: classes in after but not in before
        - resolved: classes in before but not in after
        """
        before_classes = {d.class_id for d in before if d.is_violation}
        after_classes = {d.class_id for d in after if d.is_violation}

        new_v = [d for d in after if d.is_violation and d.class_id not in before_classes]
        resolved = [d for d in before if d.is_violation and d.class_id not in after_classes]

        return new_v, resolved
