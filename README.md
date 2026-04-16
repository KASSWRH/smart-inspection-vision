# 🔍 Smart Inspection Vision System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-orange.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Production-ready AI system for visual inspection and compliance detection.**  
> Supports edge/mobile deployment (TFLite/ONNX), before/after comparison, offline-first design, and full MLOps pipeline.

---

## 📐 Architecture Overview

```
smart-inspection-vision/
├── app/
│   ├── api/              # FastAPI routers (REST endpoints)
│   ├── core/             # Config, logging, security
│   ├── models/           # Pydantic schemas & DB models
│   ├── services/         # Business logic (detection, comparison, reporting)
│   └── utils/            # Helpers (image processing, converters)
├── configs/              # YAML configs per environment
├── data/                 # Sample & processed data
├── deployment/
│   ├── docker/           # Dockerfile + docker-compose
│   └── k8s/              # Kubernetes manifests
├── docs/                 # API docs, architecture diagrams
├── notebooks/            # EDA and training experiments
├── scripts/              # Training, export, evaluation scripts
└── tests/                # Unit & integration tests
```

---

## 🚀 Key Features

| Feature | Details |
|---|---|
| **Object Detection** | YOLOv8 multi-class detection with confidence scoring |
| **Before/After Comparison** | Structural similarity + pixel-diff change detection |
| **Compliance Scoring** | Rule-based violation classification with severity levels |
| **Edge Deployment** | ONNX & TFLite export for mobile/tablet offline use |
| **Arabic + English** | Bilingual API responses and reporting |
| **MLOps** | MLflow experiment tracking, model versioning, drift detection |
| **Audit Logging** | Every AI decision logged with explainability metadata |
| **REST API** | FastAPI with OpenAPI docs, JWT auth, rate limiting |

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/your-username/smart-inspection-vision.git
cd smart-inspection-vision

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp configs/env.example .env
# Edit .env with your settings
```

### 3. Run with Docker
```bash
docker-compose -f deployment/docker/docker-compose.yml up --build
```

### 4. Run Locally (Dev)
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## 🧪 Running Tests
```bash
pytest tests/ -v --cov=app --cov-report=html
```

---

## 📦 Model Export (Edge/Mobile)
```bash
# Export to ONNX
python scripts/export_model.py --format onnx --model-path models/best.pt

# Export to TFLite (mobile)
python scripts/export_model.py --format tflite --quantize int8
```

---

## 🏗️ MLOps Pipeline
```bash
# Train with experiment tracking
python scripts/train.py --config configs/training.yaml --experiment smart-inspection-v1

# Evaluate model drift
python scripts/evaluate_drift.py --model-version v1.2

# View experiments
mlflow ui --port 5000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/inspect/detect` | Run object detection on image |
| `POST` | `/api/v1/inspect/compare` | Before/after image comparison |
| `POST` | `/api/v1/inspect/compliance` | Full compliance scoring report |
| `GET` | `/api/v1/models/info` | Active model metadata |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/audit/logs` | Retrieve AI decision audit logs |

---

## 🏛️ AI Governance

- All predictions include **confidence scores** and **explainability metadata**
- **Audit trail** stored per request (model version, timestamp, input hash, output)
- **Bias monitoring** via per-class performance tracking
- **Drift detection** compares live distribution vs training baseline
- Compliant with data privacy standards (no PII stored in audit logs)

---

## 🌍 Bilingual Support

All API responses include both Arabic and English fields:
```json
{
  "violation_type": "Improper Waste Disposal",
  "violation_type_ar": "التخلص غير السليم من النفايات",
  "severity": "high",
  "severity_ar": "عالي"
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 👤 Author

**Kasswrh AL-Mohamadi**  
AI/ML Engineer | Computer Vision & Generative AI  
Riyadh, Saudi Arabia