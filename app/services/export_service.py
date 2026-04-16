"""
Model Export Service
====================
Exports YOLOv8 models to ONNX and TFLite formats for edge/mobile deployment.
Supports INT8 quantization for reduced memory footprint.
"""

from pathlib import Path
from typing import Literal

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

ExportFormat = Literal["onnx", "tflite", "torchscript"]


class ModelExportService:
    """
    Handles model conversion to edge-compatible formats.

    Supported targets:
    - ONNX: Cross-platform, used with onnxruntime on Android/PC.
    - TFLite: Android/iOS inference via TensorFlow Lite.
    - TorchScript: On-device PyTorch Mobile.
    """

    def export(
        self,
        model_path: Path,
        export_format: ExportFormat = "onnx",
        quantize: bool = False,
        output_dir: Path | None = None,
    ) -> Path:
        """
        Export a YOLO model to the specified format.

        Args:
            model_path: Path to source .pt model.
            export_format: Target format ("onnx", "tflite", "torchscript").
            quantize: If True, apply INT8 post-training quantization.
            output_dir: Output directory (defaults to model_path parent).

        Returns:
            Path to exported model file.

        Raises:
            FileNotFoundError: If source model does not exist.
            ValueError: If unsupported format requested.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        out_dir = output_dir or settings.MODEL_DIR / "export"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting model export",
            format=export_format,
            quantize=quantize,
            source=str(model_path),
        )

        from ultralytics import YOLO

        model = YOLO(str(model_path))

        export_kwargs: dict = {
            "format": export_format,
            "imgsz": settings.IMAGE_SIZE,
            "simplify": True,
        }

        if quantize and export_format in ("tflite", "onnx"):
            export_kwargs["int8"] = True
            logger.info("INT8 quantization enabled")

        exported_path = model.export(**export_kwargs)
        final_path = Path(str(exported_path))

        logger.info("Export complete", output=str(final_path))
        return final_path

    def verify_onnx(self, onnx_path: Path) -> bool:
        """Validate ONNX model graph integrity."""
        try:
            import onnx

            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            logger.info("ONNX model validation passed", path=str(onnx_path))
            return True
        except Exception as e:
            logger.error("ONNX validation failed", error=str(e))
            return False

    def benchmark_onnx(self, onnx_path: Path, n_runs: int = 50) -> dict:
        """
        Benchmark ONNX inference latency.

        Returns:
            Dict with mean_ms, std_ms, min_ms, max_ms.
        """
        import time

        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        dummy = np.random.rand(1, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE).astype(np.float32)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            session.run(None, {input_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)

        result = {
            "mean_ms": round(float(np.mean(times)), 2),
            "std_ms": round(float(np.std(times)), 2),
            "min_ms": round(float(np.min(times)), 2),
            "max_ms": round(float(np.max(times)), 2),
            "n_runs": n_runs,
        }
        logger.info("ONNX benchmark complete", **result)
        return result
