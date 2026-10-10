"""Shared labels, image processing, and Keras runtime for LAB3."""
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

HERE = Path(__file__).resolve().parent
LABELS = ["without_mask", "with_mask", "mask_weared_incorrect"]
DISPLAY = ["Khong deo khau trang", "Deo khau trang dung", "Deo khau trang sai"]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_keras():
    # Keep downloads/caches inside LAB3 and avoid the Windows TensorFlow DLL issue.
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ.setdefault("KERAS_HOME", str(HERE / ".keras-cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl-cache"))
    import torch
    # 8 measured faster than 4 on the 16-core lab machine; more threads regresses.
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    import keras
    return keras


def image_array(path, size=224, rng=None):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = ImageOps.fit(image, (size, size), Image.Resampling.BILINEAR)
    if rng is not None:
        if rng.random() < 0.5:
            image = ImageOps.mirror(image)
        image = image.rotate(float(rng.uniform(-8, 8)), Image.Resampling.BILINEAR)
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
    # Normalization is INSIDE the exported model. Do not normalize again in clients.
    return np.asarray(image, dtype=np.float32)
