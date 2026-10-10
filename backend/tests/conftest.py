"""Shared fixtures.

The suite never needs the trained model: a stub Keras model is injected into
the real :class:`ModelService`, so batching, timing, cropping and the HTTP
contract are exercised end to end without importing Keras.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:  # works from any working directory
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.model_service import LoadedModel, ModelService  # noqa: E402
from app.schemas import LABEL_VALUES  # noqa: E402

IMAGE_SIZE = 32
PADDING = 0.18
IMAGE_PIXELS = (120, 90)
LABELS = LABEL_VALUES


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Ignore a developer's real MASK_API_* environment."""
    for key in [name for name in os.environ if name.startswith("MASK_API_")]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Config pointing at a model that does not exist, plus test limits."""
    base = Settings.from_env()
    return dataclasses.replace(
        base,
        model_path=tmp_path / "outputs" / "run1" / "mask_model.keras",
        metadata_path=tmp_path / "outputs" / "run1" / "metadata.json",
        image_size=IMAGE_SIZE,
        padding=PADDING,
        batch_size=4,
        max_faces=8,
        max_image_bytes=1_000_000,
        max_image_pixels=1_000_000,
        load_at_startup=True,
        require_model=False,
        warmup=False,
        retry_seconds=30.0,
    )


class StubModel:
    """Minimal stand-in for a Keras model."""

    def __init__(self, image_size: int = IMAGE_SIZE, units: int = 3, logits: bool = False, delay: float = 0.0):
        self.input_shape = (None, image_size, image_size, 3)
        self.output_shape = (None, units)
        self.logits = logits
        self.delay = delay
        self.calls: list[tuple[int, ...]] = []
        self.batches: list[np.ndarray] = []
        self.raise_with: Exception | None = None

    def predict(self, batch, verbose=0):
        self.calls.append(tuple(np.shape(batch)))
        self.batches.append(np.asarray(batch, dtype=np.float32))
        if self.raise_with is not None:
            raise self.raise_with
        if self.delay:
            import time

            time.sleep(self.delay)
        rows = np.full((np.shape(batch)[0], self.output_shape[-1]), 0.03, dtype=np.float32)
        rows[:, 0] = 0.90
        if self.output_shape[-1] > 2:
            rows[:, 2] = 0.07
        if self.logits:
            rows = np.log(np.clip(rows, 1e-9, None))
        return rows


def make_loaded(model, settings: Settings, **overrides) -> LoadedModel:
    fields = dict(
        model=model,
        labels=LABELS,
        image_size=settings.image_size,
        padding=settings.padding,
        source=str(settings.model_path),
        backend="stub",
        load_ms=1.234,
        loaded_at="2026-01-01T00:00:00+00:00",
    )
    fields.update(overrides)
    return LoadedModel(**fields)


def make_image_bytes(image_format: str = "PNG", size: tuple[int, int] = IMAGE_PIXELS,
                     color=(200, 40, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=image_format)
    return buffer.getvalue()


def face(face_id: str = "f1", x=0, y=0, width=40, height=40) -> dict:
    return {"id": face_id, "x": x, "y": y, "width": width, "height": height}


def grid_faces(count: int, columns: int = 4, size: int = 20) -> list[dict]:
    """Non-overlapping boxes that always fit inside the 120x90 test image."""
    rows = max(1, -(-count // columns))
    step_x = IMAGE_PIXELS[0] // columns
    step_y = IMAGE_PIXELS[1] // rows
    return [
        face(
            f"f{index}",
            (index % columns) * step_x + 1,
            (index // columns) * step_y + 1,
            min(size, step_x - 2),
            min(size, step_y - 2),
        )
        for index in range(count)
    ]


def write_weights(settings: Settings) -> Path:
    """Create the ``.keras`` placeholder so the loader gets past the file check."""
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"stub-weights")
    return settings.model_path


def post_predict(client, faces, image: bytes | None = None, filename: str = "shot.png",
                 content_type: str = "image/png", raw_faces: str | None = None):
    payload = raw_faces if raw_faces is not None else json.dumps(faces)
    return client.post(
        "/api/v1/predict",
        files={"image": (filename, image if image is not None else make_image_bytes(), content_type)},
        data={"faces": payload},
    )


def write_metadata(path: Path, **values) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"labels": list(LABELS), "image_size": IMAGE_SIZE, "face_padding": PADDING}
    metadata.update(values)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


@pytest.fixture
def stub_model() -> StubModel:
    return StubModel()


@pytest.fixture
def service(settings, stub_model) -> ModelService:
    return ModelService(settings, loaded=make_loaded(stub_model, settings))


@pytest.fixture
def client_factory(settings, stub_model):
    """Build a live app around the real service, with a stub model.

    Keyword arguments: ``overrides`` (a dataclasses.replace kwargs dict for the
    settings), ``ready=False`` to emulate a missing .keras file, plus ``model``,
    ``labels``, ``logits`` and ``delay`` knobs for the stub.
    """
    stack = ExitStack()

    def factory(overrides: dict | None = None, *, ready: bool = True, model=None,
                labels=None, logits=None, delay=None):
        active = dataclasses.replace(settings, **(overrides or {}))
        active_model = model or stub_model
        if logits is not None:
            active_model.logits = logits
        if delay is not None:
            active_model.delay = delay
        loaded = make_loaded(active_model, active, **({"labels": labels} if labels else {}))
        service = ModelService(active, loaded=loaded if ready else None)
        return stack.enter_context(TestClient(create_app(active, service)))

    yield factory
    stack.close()


@pytest.fixture
def client(client_factory):
    return client_factory()