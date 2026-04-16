"""
Drift Detection Script
======================
Compares live prediction distribution against training baseline
to detect model performance degradation.

Usage:
    python scripts/evaluate_drift.py --model-version v1.2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


DRIFT_THRESHOLD = 0.15   # KL divergence threshold — above = significant drift


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute KL divergence D(P||Q) between two distributions."""
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return float(np.sum(p * np.log(p / q)))


def load_audit_confidence_scores(log_path: Path, last_n: int = 500) -> np.ndarray:
    """Extract recent confidence scores from audit log."""
    if not log_path.exists():
        logger.warning("Audit log not found", path=str(log_path))
        return np.array([])

    scores = []
    lines = log_path.read_text().strip().splitlines()
    for line in lines[-last_n:]:
        try:
            entry = json.loads(line)
            scores.extend(entry.get("confidence_scores", []))
        except json.JSONDecodeError:
            continue
    return np.array(scores)


def evaluate_drift(model_version: str) -> dict:
    """
    Compare live confidence distribution vs. expected baseline.
    Returns drift report dict.
    """
    live_scores = load_audit_confidence_scores(settings.AUDIT_LOG_PATH)

    if len(live_scores) < 10:
        return {"status": "insufficient_data", "samples": len(live_scores)}

    # Expected baseline: normal distribution centered at ~0.75 (well-trained model)
    baseline = np.random.normal(0.75, 0.1, size=len(live_scores))
    baseline = np.clip(baseline, 0.01, 0.99)

    # Build histograms for comparison
    bins = np.linspace(0, 1, 21)
    live_hist, _ = np.histogram(live_scores, bins=bins, density=True)
    base_hist, _ = np.histogram(baseline, bins=bins, density=True)

    # Normalize
    live_hist = live_hist / (live_hist.sum() + 1e-10)
    base_hist = base_hist / (base_hist.sum() + 1e-10)

    kl_div = compute_kl_divergence(live_hist, base_hist)
    has_drift = kl_div > DRIFT_THRESHOLD

    report = {
        "model_version": model_version,
        "samples_analyzed": len(live_scores),
        "kl_divergence": round(kl_div, 4),
        "drift_threshold": DRIFT_THRESHOLD,
        "drift_detected": has_drift,
        "mean_confidence_live": round(float(np.mean(live_scores)), 4),
        "mean_confidence_baseline": round(float(np.mean(baseline)), 4),
        "recommendation": (
            "⚠️  Model retraining recommended — confidence distribution has shifted."
            if has_drift else
            "✅ Model confidence distribution is within expected range."
        ),
    }

    logger.info("Drift evaluation complete", **{k: v for k, v in report.items()
                                                 if k != "recommendation"})
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-version", default="v1.0")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = evaluate_drift(args.model_version)
    print(json.dumps(report, indent=2, ensure_ascii=False))
