"""Request and response schemas for the LAB3 predict API.

The response models are the wire contract. They are declared explicitly (and
not dumped from dicts) so that a change in behaviour shows up as a schema
mismatch rather than as a silent API break.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Label vocabulary of LAB3 in training order (``common.LABELS``).
LABEL_WITHOUT_MASK = "without_mask"
LABEL_WITH_MASK = "with_mask"
LABEL_INCORRECT = "mask_weared_incorrect"

LABEL_VALUES = (LABEL_WITHOUT_MASK, LABEL_WITH_MASK, LABEL_INCORRECT)
MaskLabel = Literal["without_mask", "with_mask", "mask_weared_incorrect"]

CONFIDENCE_DECIMALS = 6
MILLISECONDS_DECIMALS = 3


def round_number(value: float, decimals: int = MILLISECONDS_DECIMALS) -> float:
    """Round for transport; keeps ``0.0`` instead of ``-0.0``."""
    return round(float(value) + 0.0, decimals)


def as_json_number(value: float | int) -> float | int:
    """Keep integer pixels as integers so the JSON stays byte-stable."""
    number = float(value)
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


class Box(BaseModel):
    """Axis-aligned rectangle in pixels of the original (decoded) image."""

    model_config = ConfigDict(extra="forbid")

    x: float | int
    y: float | int
    width: float | int
    height: float | int


class Prediction(BaseModel):
    """One classified face. ``box`` echoes the request box it came from."""

    id: str
    label: MaskLabel
    confidence: float = Field(ge=0.0, le=1.0)
    box: Box
    inference_ms: float = Field(ge=0.0)


class ImageSize(BaseModel):
    width: int
    height: int


class PredictResponse(BaseModel):
    request_id: str
    image: ImageSize
    predictions: list[Prediction]
    total_inference_ms: float = Field(ge=0.0)


class ModelStatus(BaseModel):
    """Model readiness, reported by ``GET /health``."""

    ready: bool
    path: str
    exists: bool
    labels: list[str]
    image_size: int | None = None
    padding: float | None = None
    batch_size: int
    backend: str | None = None
    load_ms: float | None = None
    loaded_at: str | None = None
    last_attempt_at: str | None = None
    warmup_ms: float | None = None
    metadata: dict | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ready: bool
    service: str = "lab3-mask-api"
    version: str
    uptime_seconds: float = Field(ge=0.0)
    model: ModelStatus


class ErrorResponse(BaseModel):
    """FastAPI error envelope; always ``{"detail": ...}``."""

    detail: str | dict | list


class FaceRequest(BaseModel):
    """One detector box as sent by the client in the ``faces`` form field."""

    model_config = ConfigDict(extra="ignore")

    id: str
    x: float | int
    y: float | int
    width: float | int
    height: float | int

    @field_validator("id", mode="before")
    @classmethod
    def _check_id(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("id phai la chuoi")
        if not value.strip():
            raise ValueError("id khong duoc rong")
        return value

    @field_validator("x", "y", "width", "height", mode="before")
    @classmethod
    def _check_number(cls, value: object) -> float | int:
        if isinstance(value, bool):
            raise ValueError("toa do phai la so")
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            # JSON strings such as "12.5" are common from JS clients.
            try:
                return float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError as error:
                raise ValueError(f"{value!r} khong phai la so") from error
        raise ValueError("toa do phai la so")

    @field_validator("width", "height", mode="after")
    @classmethod
    def _check_size(cls, value: float | int) -> float | int:
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("width/height phai lon hon 0")
        return value

    @field_validator("x", "y", mode="after")
    @classmethod
    def _check_origin(cls, value: float | int) -> float | int:
        if not math.isfinite(float(value)):
            raise ValueError("toa do phai la so huu han")
        if float(value) < 0:
            raise ValueError("x/y khong duoc am")
        return value