"""
Model Export Script
===================
Export trained YOLOv8 model to ONNX or TFLite for edge deployment.

Usage:
    python scripts/export_model.py --format onnx
    python scripts/export_model.py --format tflite --quantize
    python scripts/export_model.py --format onnx --benchmark
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.export_service import ModelExportService

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export model for edge deployment")
    p.add_argument("--model-path", type=Path, default=settings.DETECTION_MODEL_PATH)
    p.add_argument("--format", choices=["onnx", "tflite", "torchscript"], default="onnx")
    p.add_argument("--quantize", action="store_true", help="Apply INT8 quantization")
    p.add_argument("--output-dir", type=Path, default=settings.MODEL_DIR / "export")
    p.add_argument("--benchmark", action="store_true", help="Benchmark ONNX after export")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    svc = ModelExportService()

    exported = svc.export(
        model_path=args.model_path,
        export_format=args.format,
        quantize=args.quantize,
        output_dir=args.output_dir,
    )
    print(f"✅ Exported: {exported}")

    if args.benchmark and args.format == "onnx":
        print("\n⏱  Benchmarking ONNX runtime...")
        stats = svc.benchmark_onnx(exported)
        for k, v in stats.items():
            print(f"  {k}: {v}")
