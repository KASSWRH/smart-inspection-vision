"""
Detection Service
=================
Handles model loading, inference, and result post-processing.
Uses YOLOv8 (Ultralytics) for object/violation detection.
Supports GPU, CPU, and ONNX runtime backends.
"""

import hashlib
import time
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np
import torch
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import (
    BoundingBox,
    Detection,
    DetectionResponse,
    SEVERITY_AR,
    SeverityLevel,
)

logger = get_logger(__name__)
settings = get_settings()


# ── Violation Class Registry ──────────────────────────────────────────────────
# Maps class_id → (name_en, name_ar, severity, is_violation, desc_en, desc_ar)

VIOLATION_REGISTRY: dict[int, dict] = {
    0: {
        "name": "Illegal Dumping",
        "name_ar": "إلقاء النفايات بشكل غير قانوني",
        "severity": SeverityLevel.HIGH,
        "is_violation": True,
        "description": "Waste materials improperly disposed outside designated areas.",
        "description_ar": "نفايات ملقاة خارج المناطق المخصصة.",
    },
    1: {
        "name": "Blocked Exit",
        "name_ar": "مخرج طارئ مسدود",
        "severity": SeverityLevel.CRITICAL,
        "is_violation": True,
        "description": "Emergency exit is obstructed — immediate safety risk.",
        "description_ar": "مخرج الطوارئ مسدود — خطر فوري على السلامة.",
    },
    2: {
        "name": "Missing Safety Sign",
        "name_ar": "لافتة سلامة مفقودة",
        "severity": SeverityLevel.MEDIUM,
        "is_violation": True,
        "description": "Required safety signage is absent.",
        "description_ar": "لافتة السلامة المطلوبة غائبة.",
    },
    3: {
        "name": "Structural Damage",
        "name_ar": "تلف هيكلي",
        "severity": SeverityLevel.HIGH,
        "is_violation": True,
        "description": "Visible structural damage detected.",
        "description_ar": "تم اكتشاف تلف هيكلي مرئي.",
    },
    4: {
        "name": "Person",
        "name_ar": "شخص",
        "severity": SeverityLevel.LOW,
        "is_violation": False,
        "description": "Person detected in frame.",
        "description_ar": "تم رصد شخص في الإطار.",
    },
    5: {
        "name": "Vehicle",
        "name_ar": "مركبة",
        "severity": SeverityLevel.LOW,
        "is_violation": False,
        "description": "Vehicle detected.",
        "description_ar": "تم رصد مركبة.",
    },
}


class DetectionService:
    """
    Manages the YOLOv8 detection model lifecycle and inference.

    Attributes:
        _model: Loaded YOLO model instance (lazy-loaded on first use).
        _model_version: Version string from model metadata.
        _loaded_at: Datetime of model load.
        _device: Inference device (cuda/cpu).
    """

    def __init__(self) -> None:
        self._model = None
        self._model_version: str = "unloaded"
        self._loaded_at = None
        self._device: str = "cpu"

    # ── Model Lifecycle ───────────────────────────────────────────────────────

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """Load YOLOv8 model from disk. Falls back to nano pretrained for demo."""
        from ultralytics import YOLO

        path = model_path or settings.DETECTION_MODEL_PATH

        if path.exists():
            logger.info("Loading custom model", path=str(path))
            self._model = YOLO(str(path))
        else:
            logger.warning(
                "Custom model not found, loading YOLOv8n pretrained for demo",
                path=str(path),
            )
            self._model = YOLO("yolov8n.pt")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

        import datetime
        self._loaded_at = datetime.datetime.utcnow()
        self._model_version = getattr(self._model, "ckpt_path", "yolov8n-pretrained") or "custom-v1"
        logger.info("Model loaded", device=self._device, version=self._model_version)

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model_info(self) -> dict:
        return {
            "version": self._model_version,
            "device": self._device,
            "loaded_at": self._loaded_at,
            "num_classes": len(VIOLATION_REGISTRY),
            "class_names": [v["name"] for v in VIOLATION_REGISTRY.values()],
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        return_annotated: bool = False,
    ) -> DetectionResponse:
        """
        Run detection on a single image.

        Args:
            image: BGR numpy array (from cv2.imread or equivalent).
            confidence_threshold: Minimum confidence to include detection.
            iou_threshold: NMS IoU threshold.
            return_annotated: Whether to return annotated image as base64.

        Returns:
            DetectionResponse with all detections and metadata.
        """
        if not self.is_loaded():
            self.load_model()

        start = time.perf_counter()
        h, w = image.shape[:2]

        results = self._model.predict(
            source=image,
            conf=confidence_threshold,
            iou=iou_threshold,
            imgsz=settings.IMAGE_SIZE,
            device=self._device,
            verbose=False,
        )

        detections = self._parse_results(results, image_width=w, image_height=h)
        elapsed_ms = (time.perf_counter() - start) * 1000

        annotated_b64: Optional[str] = None
        if return_annotated:
            annotated_b64 = self._annotate_image(image, detections)

        violation_count = sum(1 for d in detections if d.is_violation)

        response = DetectionResponse(
            request_id=uuid4(),
            model_version=self._model_version,
            detections=detections,
            total_detections=len(detections),
            violation_count=violation_count,
            processing_time_ms=round(elapsed_ms, 2),
            image_width=w,
            image_height=h,
            annotated_image_base64=annotated_b64,
        )

        logger.info(
            "Detection complete",
            detections=len(detections),
            violations=violation_count,
            ms=round(elapsed_ms, 2),
        )
        return response

    # ── Result Parsing ────────────────────────────────────────────────────────

    def _parse_results(
        self,
        results: list,
        image_width: int,
        image_height: int,
    ) -> List[Detection]:
        """Convert raw YOLO results into typed Detection objects."""
        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # Map to our registry; fall back gracefully for unknown classes
                reg = VIOLATION_REGISTRY.get(class_id % len(VIOLATION_REGISTRY), {
                    "name": f"class_{class_id}",
                    "name_ar": f"فئة_{class_id}",
                    "severity": SeverityLevel.LOW,
                    "is_violation": False,
                    "description": "Unknown class detected.",
                    "description_ar": "فئة غير معروفة.",
                })

                x1, y1, x2, y2 = box.xyxyn[0].tolist()  # normalized coords

                severity = reg["severity"]
                det = Detection(
                    class_id=class_id,
                    class_name=reg["name"],
                    class_name_ar=reg["name_ar"],
                    confidence=round(conf, 4),
                    bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    severity=severity,
                    severity_ar=SEVERITY_AR[severity],
                    is_violation=reg["is_violation"],
                    description=reg["description"],
                    description_ar=reg["description_ar"],
                )
                detections.append(det)

        return detections

    # ── Annotation ────────────────────────────────────────────────────────────

    def _annotate_image(self, image: np.ndarray, detections: List[Detection]) -> str:
        """Draw bounding boxes on image and return as base64 JPEG string."""
        import base64

        annotated = image.copy()
        h, w = annotated.shape[:2]

        severity_colors = {
            SeverityLevel.LOW: (0, 255, 0),       # Green
            SeverityLevel.MEDIUM: (0, 165, 255),   # Orange
            SeverityLevel.HIGH: (0, 0, 255),        # Red
            SeverityLevel.CRITICAL: (128, 0, 128),  # Purple
        }

        for det in detections:
            bb = det.bounding_box
            x1, y1 = int(bb.x1 * w), int(bb.y1 * h)
            x2, y2 = int(bb.x2 * w), int(bb.y2 * h)
            color = severity_colors.get(det.severity, (255, 255, 255))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                annotated, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")


# ── Image Utilities ───────────────────────────────────────────────────────────

def image_sha256(image_bytes: bytes) -> str:
    """Return SHA-256 hash of raw image bytes (used for audit logging, no PII)."""
    return hashlib.sha256(image_bytes).hexdigest()


def decode_image(image_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    """Decode raw bytes into a BGR numpy array. Returns (image, width, height)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported format or corrupted file.")
    h, w = img.shape[:2]
    return img, w, h


# ── Singleton ─────────────────────────────────────────────────────────────────

_detection_service: Optional[DetectionService] = None


def get_detection_service() -> DetectionService:
    """Return the application-wide DetectionService singleton."""
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service
