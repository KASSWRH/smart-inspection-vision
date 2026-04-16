"""
Training Script
===============
Train YOLOv8 with MLflow experiment tracking, configurable hyperparameters,
and automatic model registration on completion.

Usage:
    python scripts/train.py --config configs/training.yaml
    python scripts/train.py --experiment my-exp --epochs 100 --imgsz 640
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import mlflow.pytorch
import yaml

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 with MLflow tracking")
    p.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    p.add_argument("--experiment", type=str, default=settings.MLFLOW_EXPERIMENT_NAME)
    p.add_argument("--data", type=str, help="Path to dataset YAML")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Base model")
    p.add_argument("--name", type=str, default="inspection-run")
    return p.parse_args()


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    logger.warning("Config file not found — using defaults", path=str(config_path))
    return {}


def train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    config = load_config(args.config)
    train_cfg = config.get("training", {})

    # CLI args override config file
    epochs = args.epochs or train_cfg.get("epochs", 50)
    imgsz = args.imgsz or train_cfg.get("imgsz", 640)
    batch = args.batch or train_cfg.get("batch", 16)
    data = args.data or train_cfg.get("data", "data/dataset.yaml")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.name) as run:
        logger.info("MLflow run started", run_id=run.info.run_id)

        # Log hyperparameters
        mlflow.log_params({
            "model_base": args.model,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "data": data,
        })

        # Train
        model = YOLO(args.model)
        results = model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=args.name,
            exist_ok=True,
        )

        # Log metrics
        if results and hasattr(results, "results_dict"):
            for k, v in results.results_dict.items():
                try:
                    mlflow.log_metric(k.replace("/", "_"), float(v))
                except (TypeError, ValueError):
                    pass

        # Save best model artifact
        best_pt = Path(f"runs/detect/{args.name}/weights/best.pt")
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="model")
            logger.info("Best model logged to MLflow", path=str(best_pt))

        logger.info("Training complete", run_id=run.info.run_id)


if __name__ == "__main__":
    args = parse_args()
    train(args)
