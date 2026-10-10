"""End-to-end tests that really load a ``.keras`` file.

They need Keras + torch, so they are marked ``keras`` and skipped when the
runtime is unavailable:

    python -m pytest -m keras
"""

from __future__ import annotations

import dataclasses
import io
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.model_service import ModelService, import_keras
from conftest import face, post_predict, write_metadata

pytestmark = pytest.mark.keras

LABELS = {"without_mask", "with_mask", "mask_weared_incorrect"}


@pytest.fixture(scope="module")
def keras_module():
    try:
        return import_keras()
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"Keras khong san sang: {error}")


def save_model(path, keras_module, image_size: int = 32):
    """Write a tiny but *real* .keras model with the LAB3 tensor contract."""
    keras_module.utils.set_random_seed(42)
    inputs = keras_module.Input(shape=(image_size, image_size, 3), name="rgb_pixels_0_255")
    x = keras_module.layers.Rescaling(1 / 127.5, offset=-1, name="normalize")(inputs)
    x = keras_module.layers.GlobalAveragePooling2D()(x)
    outputs = keras_module.layers.Dense(3, activation="softmax", name="mask_classes")(x)
    model = keras_module.Model(inputs, outputs, name="tiny_lab3_mask")

    # Brightness decides the class, so the assertions below are deterministic.
    samples = np.zeros((4, image_size, image_size, 3), dtype=np.float32)
    samples[2:] = 255.0
    targets = np.eye(3, dtype=np.float32)[[0, 0, 2, 2]]
    model.compile(optimizer=keras_module.optimizers.Adam(0.1),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(samples, targets, batch_size=4, epochs=100, verbose=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return model


def two_tone_image() -> bytes:
    """Left half black, right half white, so the two crops differ."""
    buffer = io.BytesIO()
    picture = Image.new("RGB", (120, 90), (0, 0, 0))
    picture.paste((255, 255, 255), (60, 0, 120, 90))
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def test_real_keras_file_is_loaded_and_serves_predictions(tmp_path, settings, keras_module):
    model_path = tmp_path / "outputs" / "run1" / "mask_model.keras"
    metadata_path = tmp_path / "outputs" / "run1" / "metadata.json"
    save_model(model_path, keras_module)
    write_metadata(metadata_path, image_size=32, face_padding=0.18)
    settings = dataclasses.replace(settings, model_path=model_path, metadata_path=metadata_path)
    service = ModelService(settings)

    with TestClient(create_app(settings, service)) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["model"]["ready"] is True
        assert health["model"]["image_size"] == 32
        assert health["model"]["padding"] == 0.18
        assert health["model"]["exists"] is True
        assert health["model"]["metadata"]["labels"] == ["without_mask", "with_mask", "mask_weared_incorrect"]

        body = post_predict(
            client,
            [face("dark", 0, 10, 40, 40), face("bright", 80, 10, 40, 40)],
            image=two_tone_image(),
        ).json()

    assert [p["id"] for p in body["predictions"]] == ["dark", "bright"]
    assert all(p["label"] in LABELS for p in body["predictions"])
    assert all(p["confidence"] > 0.0 for p in body["predictions"])
    assert body["total_inference_ms"] > 0.0
    assert sum(p["inference_ms"] for p in body["predictions"]) == pytest.approx(
        body["total_inference_ms"], abs=0.01)
    assert body["predictions"][0]["label"] != body["predictions"][1]["label"], (
        "the tiny model should separate a black crop from a white one"
    )


def test_model_path_environment_variable_drives_the_loader(tmp_path, monkeypatch, keras_module):
    model_path = tmp_path / "nested" / "custom.keras"
    save_model(model_path, keras_module)
    monkeypatch.setenv("MASK_API_MODEL_PATH", str(model_path))
    monkeypatch.setenv("MASK_API_WARMUP", "false")

    service = ModelService(Settings.from_env())
    loaded = service.load()

    assert loaded.source == str(model_path)
    assert loaded.image_size == 32
    assert service.status()["ready"] is True


def test_default_settings_point_at_the_trained_model(monkeypatch):
    """The documented default is outputs/run1/mask_model.keras."""
    for name in [key for key in os.environ if key.startswith("MASK_API_")]:
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.model_path.parts[-3:] == ("outputs", "run1", "mask_model.keras")
    assert settings.metadata_path.name == "metadata.json"