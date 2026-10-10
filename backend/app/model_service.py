"""Keras model loading, readiness reporting and batched inference.

The service never imports Keras at module import time, so ``app.main`` (and the
test suite) can be imported on a machine where the weights are missing or
TensorFlow/Keras is not installed yet. Keras is imported on first use and the
result is cached.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import (
    PROJECT_ROOT,
    Settings,
    metadata_image_size,
    metadata_labels,
    metadata_padding,
    read_metadata,
)
from .errors import ModelNotReadyError
from .preprocessing import FaceBox, build_batch
from .schemas import LABEL_VALUES, round_number

LOGGER = logging.getLogger("lab3.model")

SOFTMAX_TOLERANCE = 1e-3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def import_keras() -> Any:
    """Import Keras with the same environment LAB3 training uses.

    ``common.load_keras`` pins ``KERAS_BACKEND=torch`` and keeps the Keras
    cache inside the project, which also avoids the Windows DLL issue. The
    fallback keeps the API usable when only ``backend/`` is deployed.
    """
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from common import load_keras  # noqa: PLC0415  (optional project dependency)

        return load_keras()
    except Exception as error:  # noqa: BLE001 - any failure falls back
        LOGGER.debug("common.load_keras khong dung duoc (%s), dung keras truc tiep.", error)
        os.environ.setdefault("KERAS_BACKEND", "torch")
        import keras  # noqa: PLC0415

        return keras


@dataclass
class LoadedModel:
    """Everything inference needs, resolved once at load time."""

    model: Any
    labels: tuple[str, ...]
    image_size: int
    padding: float
    source: str
    backend: str | None
    load_ms: float
    loaded_at: str
    warmup_ms: float | None = None
    metadata: dict | None = None

    def describe(self) -> dict:
        return {
            "source": self.source,
            "labels": list(self.labels),
            "image_size": self.image_size,
            "padding": self.padding,
            "backend": self.backend,
            "load_ms": round_number(self.load_ms, 1),
            "loaded_at": self.loaded_at,
            "warmup_ms": round_number(self.warmup_ms, 1) if self.warmup_ms is not None else None,
        }


@dataclass
class FaceOutcome:
    """Result of classifying one face."""

    index: int
    label: str
    confidence: float
    probabilities: list[float]
    inference_ms: float


class ModelService:
    """Thread-safe owner of the Keras model and its readiness state."""

    def __init__(self, settings: Settings, loaded: LoadedModel | None = None) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._loaded: LoadedModel | None = loaded
        self._error: str | None = None
        self._last_attempt_at: float | None = None
        self._last_attempt_iso: str | None = None

    # ------------------------------------------------------------------ state

    @property
    def ready(self) -> bool:
        return self._loaded is not None

    @property
    def model(self) -> Any:
        return self._loaded.model if self._loaded else None

    def status(self) -> dict:
        """Readiness payload for ``GET /health``."""
        loaded = self._loaded
        path = str(self.settings.model_path)
        return {
            "ready": loaded is not None,
            "path": loaded.source if loaded else path,
            "exists": self.settings.model_path.is_file(),
            "labels": list(loaded.labels) if loaded else list(self.settings.labels),
            "image_size": loaded.image_size if loaded else None,
            "padding": loaded.padding if loaded else None,
            "batch_size": self.settings.batch_size,
            "backend": loaded.backend if loaded else None,
            "load_ms": round_number(loaded.load_ms, 1) if loaded else None,
            "loaded_at": loaded.loaded_at if loaded else None,
            "last_attempt_at": self._last_attempt_iso,
            "warmup_ms": round_number(loaded.warmup_ms, 1) if loaded and loaded.warmup_ms else None,
            "metadata": loaded.metadata if loaded else read_metadata(self.settings.metadata_path),
            "error": None if loaded else self._error,
        }

    # ----------------------------------------------------------------- loading

    def load(self, force: bool = False) -> LoadedModel:
        """Load the model if needed; raise :class:`ModelNotReadyError`."""
        if self._loaded is not None and not force:
            return self._loaded
        with self._lock:
            if self._loaded is not None and not force:
                return self._loaded
            self._last_attempt_at = time.monotonic()
            self._last_attempt_iso = _now()
            try:
                loaded = self._load_locked()
            except Exception as error:  # noqa: BLE001 - surfaced as 503, never a crash
                self._error = str(error) or error.__class__.__name__
                LOGGER.error("Khong nap duoc model %s: %s", self.settings.model_path, self._error)
                raise ModelNotReadyError(self._error) from error
            self._loaded = loaded
            self._error = None
            LOGGER.info(
                "Model san sang: %s | nhan=%s | image_size=%s | padding=%s | backend=%s | load=%.1fms",
                loaded.source, ",".join(loaded.labels), loaded.image_size,
                loaded.padding, loaded.backend, loaded.load_ms,
            )
            return loaded

    def _load_locked(self) -> LoadedModel:
        path: Path = self.settings.model_path
        if not path.is_file():
            raise ModelNotReadyError(
                f"Khong tim thay model tai {path}. Chay train.py hoac dat MASK_API_MODEL_PATH."
            )
        metadata = read_metadata(self.settings.metadata_path)
        started = time.perf_counter()
        keras = import_keras()
        try:
            model = keras.models.load_model(path, compile=False)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"Khong doc duoc file .keras {path}: {error}") from error
        load_ms = (time.perf_counter() - started) * 1000.0

        labels = self._resolve_labels(model, metadata)
        image_size = self._resolve_image_size(model, metadata)
        padding = metadata_padding(metadata)
        if padding is None:
            padding = self.settings.padding
        backend = getattr(getattr(keras, "backend", None), "backend", lambda: None)()
        loaded = LoadedModel(
            model=model,
            labels=labels,
            image_size=image_size,
            padding=float(padding),
            source=str(path),
            backend=backend,
            load_ms=load_ms,
            loaded_at=_now(),
            metadata=metadata,
        )
        if self.settings.warmup:
            loaded.warmup_ms = self._warmup(loaded)
        return loaded

    def _resolve_labels(self, model: Any, metadata: dict | None) -> tuple[str, ...]:
        candidates = [metadata_labels(metadata), tuple(self.settings.labels)]
        for labels in candidates:
            if not labels:
                continue
            outputs = self._output_units(model)
            if outputs is not None and outputs != len(labels):
                continue
            unknown = [label for label in labels if label not in LABEL_VALUES]
            if unknown:
                raise RuntimeError(
                    f"Nhan khong hop le {unknown}. LAB3 chi dung {list(LABEL_VALUES)}."
                )
            return tuple(labels)
        raise RuntimeError(
            f"Model xuat khong dung so nhan. Can 3 dau ra theo thu tu {list(LABEL_VALUES)}."
        )

    def _resolve_image_size(self, model: Any, metadata: dict | None) -> int:
        shape = getattr(model, "input_shape", None)
        size = metadata_image_size(metadata) or self.settings.image_size
        if isinstance(shape, list) and shape:
            shape = shape[0]
        if isinstance(shape, (tuple, list)) and len(shape) == 4:
            height, width = shape[1], shape[2]
            if isinstance(height, int) and isinstance(width, int) and height == width and height > 0:
                return height
            raise RuntimeError(f"Model dau vao khong vuong: {shape}.")
        LOGGER.warning("Khong doc duoc input_shape cua model; dung image_size=%s.", size)
        return size

    @staticmethod
    def _output_units(model: Any) -> int | None:
        shape = getattr(model, "output_shape", None)
        if isinstance(shape, list) and shape:
            shape = shape[0]
        if isinstance(shape, (tuple, list)) and len(shape) == 2 and isinstance(shape[-1], int):
            return shape[-1]
        return None

    def _warmup(self, loaded: LoadedModel) -> float:
        try:
            dummy = np.zeros((1, loaded.image_size, loaded.image_size, 3), np.float32)
            started = time.perf_counter()
            loaded.model.predict(dummy, verbose=0)
            return (time.perf_counter() - started) * 1000.0
        except Exception as error:  # noqa: BLE001 - warmup is best effort
            LOGGER.warning("Warmup that bai: %s", error)
            return 0.0

    def ensure_ready(self) -> LoadedModel:
        """Return the model or raise; retries lazily after ``retry_seconds``."""
        if self._loaded is not None:
            return self._loaded
        now = time.monotonic()
        if self._last_attempt_at is None or now - self._last_attempt_at >= self.settings.retry_seconds:
            return self.load()
        raise ModelNotReadyError(self._error or "Model chua san sang.")

    # --------------------------------------------------------------- inference

    def classify(self, image: Any, faces: Sequence[FaceBox]) -> tuple[list[FaceOutcome], float]:
        """Classify every face; returns per-face outcomes and total ms.

        Faces are pushed through the model in batches of
        ``settings.batch_size``. Each face is charged an equal share of its
        batch wall time, so ``sum(inference_ms) == total_inference_ms``.
        """
        loaded = self.ensure_ready()
        if not faces:
            return [], 0.0
        batch = build_batch(image, faces, loaded.image_size, loaded.padding)
        outcomes: list[FaceOutcome] = []
        total_ms = 0.0
        size = self.settings.batch_size
        for start in range(0, len(faces), size):
            chunk = batch[start:start + size]
            started = time.perf_counter()
            raw = loaded.model.predict(chunk, verbose=0)
            elapsed = (time.perf_counter() - started) * 1000.0
            probabilities = _probabilities(loaded.model, raw, len(loaded.labels))
            if probabilities.shape[0] != chunk.shape[0]:
                raise RuntimeError(
                    f"Model tra ve {probabilities.shape[0]} du doan cho {chunk.shape[0]} khuon mat."
                )
            per_face = elapsed / chunk.shape[0]
            for offset, row in enumerate(probabilities):
                index = int(np.argmax(row))
                outcomes.append(FaceOutcome(
                    index=start + offset,
                    label=loaded.labels[index],
                    confidence=round_number(float(row[index]), 6),
                    probabilities=[round_number(float(value), 6) for value in row],
                    inference_ms=round_number(per_face, 3),
                ))
            total_ms += elapsed
        return outcomes, round_number(total_ms, 3)


def _probabilities(model: Any, raw: Any, expected: int | None = None) -> np.ndarray:
    """Coerce a model output into a ``(n, classes)`` probability matrix."""
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise RuntimeError(f"Model tra ve shape khong mong doi: {values.shape}.")
    if expected is not None and values.shape[1] != expected:
        raise RuntimeError(f"Model tra ve {values.shape[1]} lop, can {expected}.")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Model tra ve gia tri khong huu han.")
    totals = values.sum(axis=1, keepdims=True)
    if np.all(values >= -1e-6) and np.allclose(totals, 1.0, atol=SOFTMAX_TOLERANCE):
        return _normalize(values)
    return _softmax(values)


def _normalize(values: np.ndarray) -> np.ndarray:
    totals = np.clip(values.sum(axis=1, keepdims=True), 1e-12, None)
    return np.clip(values / totals, 0.0, 1.0)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)
