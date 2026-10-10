"""Runtime configuration for the LAB3 mask-classification API.

Everything is driven by environment variables so the same code runs in a
container, from pytest, or straight from a checkout. Relative paths resolve
against the LAB3 project root (the parent of ``backend/``), never against the
process working directory.

===========================  ==========================================
Environment variable          Meaning / default
===========================  ==========================================
``MASK_API_MODEL_PATH``      ``.keras`` weights. Default
                             ``outputs/run1/mask_model.keras``.
``MASK_API_METADATA_PATH``   ``metadata.json`` written by ``train.py``.
                             Default: ``<model dir>/metadata.json``.
``MASK_API_LABELS``          Comma separated label order. Default comes
                             from ``common.LABELS``.
``MASK_API_IMAGE_SIZE``      Fallback square input size when the model
                             does not expose a shape. Default ``224``.
``MASK_API_PADDING``         Fallback face padding. Default ``0.18``
                             (``prepare_dataset.py`` default, also stored
                             as ``face_padding`` in ``metadata.json``).
``MASK_API_BATCH_SIZE``      Faces per ``model.predict`` call. Default 16.
``MASK_API_MAX_FACES``       Hard cap on faces per request. Default 64.
``MASK_API_MAX_IMAGE_BYTES`` Uploaded image cap in bytes. Default 10 MiB.
``MASK_API_MAX_IMAGE_PIXELS`` Decoded image cap in pixels. Default 40 MP.
``MASK_API_CONTENT_TYPES``   Accepted upload MIME types. Default
                             ``image/jpeg,image/png``.
``MASK_API_LOAD_AT_STARTUP`` Try to load the model during startup.
                             Default ``true``.
``MASK_API_REQUIRE_MODEL``   Fail startup instead of serving a degraded
                             service. Default ``false``.
``MASK_API_WARMUP``          Run one dummy forward pass after load so the
                             first request is not penalised. Default
                             ``true``.
``MASK_API_RETRY_SECONDS``   Minimum delay before a failed lazy load is
                             retried. Default 30 seconds.
``MASK_API_LOG_LEVEL``       ``logging`` level name. Default ``INFO``.
===========================  ==========================================
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger("lab3.config")

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

DEFAULT_MODEL_PATH = Path("outputs") / "run1" / "mask_model.keras"
DEFAULT_LABELS = ("without_mask", "with_mask", "mask_weared_incorrect")
DEFAULT_PADDING = 0.18
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CONTENT_TYPES = ("image/jpeg", "image/png")

#: Content types that map onto the same decoder; ``image/jpg`` is a common
#: (non standard) spelling sent by browsers and curl scripts.
CONTENT_TYPE_ALIASES = {"image/jpg": "image/jpeg", "image/pjpeg": "image/jpeg"}

TRUE_WORDS = {"1", "true", "yes", "on", "y", "t"}
FALSE_WORDS = {"0", "false", "no", "off", "n", "f"}


class ConfigurationError(RuntimeError):
    """Raised when an environment variable cannot be used."""


def _raw(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def env_str(name: str, default: str) -> str:
    value = _raw(name)
    return default if value is None else value


def env_path(name: str, default: Path | str | None) -> Path | None:
    value = _raw(name)
    if value is None:
        return Path(default) if default is not None else None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def env_int(name: str, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    value = _raw(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} phai la so nguyen, nhan {value!r}") from error
    if parsed < minimum or (maximum is not None and parsed > maximum):
        raise ConfigurationError(f"{name} phai nam trong [{minimum}, {maximum or 'inf'}]")
    return parsed


def env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = _raw(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} phai la so thuc, nhan {value!r}") from error
    if not minimum <= parsed <= maximum:
        raise ConfigurationError(f"{name} phai nam trong [{minimum}, {maximum}]")
    return parsed


def env_bool(name: str, default: bool) -> bool:
    value = _raw(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in TRUE_WORDS:
        return True
    if lowered in FALSE_WORDS:
        return False
    raise ConfigurationError(f"{name} phai la true/false, nhan {value!r}")


def env_labels(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = _raw(name)
    if value is None:
        return tuple(default)
    labels = tuple(part.strip() for part in value.split(",") if part.strip())
    if not labels:
        raise ConfigurationError(f"{name} rong")
    return labels


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the service configuration."""

    model_path: Path
    metadata_path: Path | None
    labels: tuple[str, ...]
    image_size: int
    padding: float
    batch_size: int
    max_faces: int
    max_image_bytes: int
    max_image_pixels: int
    content_types: tuple[str, ...]
    load_at_startup: bool
    require_model: bool
    warmup: bool
    retry_seconds: float
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        model_path = env_path("MASK_API_MODEL_PATH", DEFAULT_MODEL_PATH)
        metadata = env_path("MASK_API_METADATA_PATH", None)
        if metadata is None:
            metadata = model_path.parent / "metadata.json"
        content_types = env_labels("MASK_API_CONTENT_TYPES", DEFAULT_CONTENT_TYPES)
        settings = cls(
            model_path=model_path,
            metadata_path=metadata,
            labels=env_labels("MASK_API_LABELS", DEFAULT_LABELS),
            image_size=env_int("MASK_API_IMAGE_SIZE", DEFAULT_IMAGE_SIZE, minimum=1, maximum=4096),
            padding=env_float("MASK_API_PADDING", DEFAULT_PADDING, 0.0, 0.5),
            batch_size=env_int("MASK_API_BATCH_SIZE", 16, minimum=1, maximum=512),
            max_faces=env_int("MASK_API_MAX_FACES", 64, minimum=1, maximum=4096),
            max_image_bytes=env_int("MASK_API_MAX_IMAGE_BYTES", 10 * 1024 * 1024, minimum=1),
            max_image_pixels=env_int("MASK_API_MAX_IMAGE_PIXELS", 40_000_000, minimum=1),
            content_types=content_types,
            load_at_startup=env_bool("MASK_API_LOAD_AT_STARTUP", True),
            require_model=env_bool("MASK_API_REQUIRE_MODEL", False),
            warmup=env_bool("MASK_API_WARMUP", True),
            retry_seconds=env_float("MASK_API_RETRY_SECONDS", 30.0, 0.0, 86400.0),
            log_level=env_str("MASK_API_LOG_LEVEL", "INFO").upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if len(self.labels) < 2:
            raise ConfigurationError("Can it nhat 2 nhan.")
        if len(set(self.labels)) != len(self.labels):
            raise ConfigurationError("Cac nhan phai khac nhau.")
        if len(self.content_types) == 0:
            raise ConfigurationError("Can it nhat mot MIME type.")

    def canonical_content_type(self, content_type: str | None) -> str | None:
        """Map an upload MIME type onto the decoder key, or ``None``."""
        if not content_type:
            return None
        base = content_type.split(";")[0].strip().lower()
        base = CONTENT_TYPE_ALIASES.get(base, base)
        return base if base in self.content_types else None

    def as_public_dict(self) -> dict:
        """Values safe to echo in logs and ``/health``."""
        return {
            "model_path": str(self.model_path),
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "labels": list(self.labels),
            "image_size": self.image_size,
            "padding": self.padding,
            "batch_size": self.batch_size,
            "max_faces": self.max_faces,
            "max_image_bytes": self.max_image_bytes,
            "max_image_pixels": self.max_image_pixels,
            "content_types": list(self.content_types),
            "load_at_startup": self.load_at_startup,
            "require_model": self.require_model,
            "warmup": self.warmup,
        }


def read_metadata(path: Path | None) -> dict | None:
    """Load ``metadata.json`` written by ``train.py``; never raise."""
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        LOGGER.warning("Khong doc duoc metadata %s: %s", path, error)
        return None
    if not isinstance(payload, dict):
        LOGGER.warning("Bo qua metadata %s: khong phai JSON object.", path)
        return None
    return payload


def metadata_labels(metadata: dict | None) -> tuple[str, ...] | None:
    if not metadata:
        return None
    labels = metadata.get("labels")
    if isinstance(labels, list) and labels and all(isinstance(item, str) for item in labels):
        return tuple(labels)
    return None


def metadata_padding(metadata: dict | None) -> float | None:
    """Face padding used when ``prepare_dataset.py`` built the crops."""
    if not metadata:
        return None
    candidates = [metadata.get("face_padding")]
    preparation = metadata.get("dataset_preparation")
    if isinstance(preparation, dict):
        candidates.append(preparation.get("padding"))
    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            continue
        value = float(candidate)
        if 0.0 <= value <= 0.5:
            return value
    return None


def metadata_image_size(metadata: dict | None) -> int | None:
    if not metadata:
        return None
    size = metadata.get("image_size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        return None
    return size