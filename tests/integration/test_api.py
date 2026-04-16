"""
Integration Tests — API Endpoints
==================================
Uses FastAPI TestClient with mocked detection service.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import DetectionResponse, SeverityLevel
from app.services.detection_service import get_detection_service


def _make_test_jpeg() -> bytes:
    import cv2
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_detector():
    mock = MagicMock()
    mock.is_loaded.return_value = True
    mock.get_model_info.return_value = {
        "version": "test-v1",
        "device": "cpu",
        "loaded_at": None,
        "num_classes": 6,
        "class_names": ["cls0", "cls1", "cls2", "cls3", "cls4", "cls5"],
    }
    mock.detect.return_value = DetectionResponse(
        model_version="test-v1",
        detections=[],
        total_detections=0,
        violation_count=0,
        processing_time_ms=12.0,
        image_width=128,
        image_height=128,
    )
    return mock


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestDetectEndpoint:
    def test_detect_valid_image(self, client, mock_detector):
        app.dependency_overrides[get_detection_service] = lambda: mock_detector
        jpeg = _make_test_jpeg()
        response = client.post(
            "/api/v1/inspect/detect",
            files={"image": ("test.jpg", io.BytesIO(jpeg), "image/jpeg")},
            data={"confidence_threshold": "0.45"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "model_version" in data
        app.dependency_overrides.clear()

    def test_detect_invalid_file(self, client):
        response = client.post(
            "/api/v1/inspect/detect",
            files={"image": ("bad.jpg", io.BytesIO(b"not-an-image"), "image/jpeg")},
        )
        assert response.status_code == 422

    def test_detect_returns_request_id(self, client, mock_detector):
        app.dependency_overrides[get_detection_service] = lambda: mock_detector
        jpeg = _make_test_jpeg()
        response = client.post(
            "/api/v1/inspect/detect",
            files={"image": ("test.jpg", io.BytesIO(jpeg), "image/jpeg")},
        )
        assert "request_id" in response.json()
        app.dependency_overrides.clear()


class TestAuditEndpoint:
    def test_audit_logs_returns_list(self, client):
        response = client.get("/api/v1/audit/logs?limit=5")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
