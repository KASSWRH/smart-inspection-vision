"""
Image Processing Utilities
===========================
Helper functions for image validation, resizing, and format conversion.
"""

import base64
from typing import Tuple

import cv2
import numpy as np


def resize_with_aspect(
    image: np.ndarray,
    target_size: int = 640,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Letterbox-resize image to target_size × target_size while maintaining aspect ratio.

    Returns:
        (resized_image, scale_factor, (pad_w, pad_h))
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=pad_color)
    return padded, scale, (pad_w // 2, pad_h // 2)


def numpy_to_base64(image: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR numpy image as base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def validate_image_size(width: int, height: int, min_px: int = 32, max_px: int = 8192) -> bool:
    """Validate that image dimensions are within acceptable bounds."""
    return all(min_px <= dim <= max_px for dim in (width, height))
