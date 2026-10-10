"""Crop geometry and configuration helpers.

``reference_crop_bounds`` is a verbatim copy of the crop maths inside
``prepare_dataset.prepare`` (LAB3 root). If that script changes, this test is
the canary that says the API no longer frames faces like the training crops.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from app.config import (
    ConfigurationError,
    Settings,
    metadata_image_size,
    metadata_labels,
    metadata_padding,
)
from app.preprocessing import (
    FaceBox,
    build_batch,
    crop_face,
    face_crop_bounds,
    sniff_content_type,
)

IMAGE = (640, 480)


def reference_crop_bounds(x1, y1, x2, y2, width, height, padding):
    """Copy of prepare_dataset.prepare lines 106-110."""
    side = min(max(x2 - x1, y2 - y1) * (1 + 2 * padding), width, height)
    left = max(0, min(width - side, (x1 + x2 - side) / 2))
    top = max(0, min(height - side, (y1 + y2 - side) / 2))
    return (int(left), int(top), min(width, math.ceil(left + side)), min(height, math.ceil(top + side)))


CASES = [
    # (x, y, w, h) inside a 640x480 image
    (100, 100, 50, 50),
    (0, 0, 50, 50),            # top-left corner -> clamped
    (590, 430, 50, 50),        # bottom-right corner -> clamped
    (0, 200, 640, 80),         # full width
    (300, 0, 40, 480),         # full height
    (10, 10, 1, 1),            # tiny
    (0, 0, 640, 480),          # whole image
    (100, 100, 500, 40),       # wide, not square
    (100, 100, 40, 500),       # tall, not square
    (37, 91, 13, 211),         # odd numbers, rounding
]


@pytest.mark.parametrize("padding", [0.0, 0.1, 0.18, 0.5])
@pytest.mark.parametrize("x, y, w, h", CASES)
def test_crop_bounds_match_prepare_dataset(padding, x, y, w, h):
    box = FaceBox(id="f", x=float(x), y=float(y), width=float(w), height=float(h))

    assert face_crop_bounds(box, IMAGE, padding) == reference_crop_bounds(x, y, x + w, y + h, *IMAGE, padding)


@pytest.mark.parametrize("padding", [0.0, 0.18])
def test_crop_bounds_match_prepare_dataset_for_fractional_boxes(padding):
    box = FaceBox(id="f", x=10.5, y=20.25, width=30.5, height=15.75)

    assert face_crop_bounds(box, IMAGE, padding) == reference_crop_bounds(
        10.5, 20.25, 41.0, 36.0, *IMAGE, padding)


def test_zero_padding_still_squares_the_box_on_its_longest_side():
    """Padding widens the square; it never distorts it (VOC crops are square)."""
    box = FaceBox(id="f", x=100, y=120, width=50, height=70)

    left, top, right, bottom = face_crop_bounds(box, IMAGE, 0.0)

    assert (right - left, bottom - top) == (70, 70)
    assert (left + right) / 2 == pytest.approx(box.x + box.width / 2, abs=1)
    assert (top + bottom) / 2 == pytest.approx(box.y + box.height / 2, abs=1)


def test_padding_expands_around_the_centre():
    box = FaceBox(id="f", x=200, y=200, width=100, height=100)

    unpadded = face_crop_bounds(box, IMAGE, 0.0)
    padded = face_crop_bounds(box, IMAGE, 0.18)

    assert padded[2] - padded[0] > unpadded[2] - unpadded[0]
    assert abs(((padded[0] + padded[2]) / 2) - ((unpadded[0] + unpadded[2]) / 2)) <= 1


def test_crops_never_leave_the_image():
    for x, y, w, h in CASES:
        box = FaceBox(id="f", x=float(x), y=float(y), width=float(w), height=float(h))

        left, top, right, bottom = face_crop_bounds(box, IMAGE, 0.18)

        assert 0 <= left < right <= IMAGE[0]
        assert 0 <= top < bottom <= IMAGE[1]
        assert abs((right - left) - (bottom - top)) <= 1  # square up to rounding


def test_build_batch_stacks_square_float32_crops():
    picture = Image.new("RGB", IMAGE, (120, 130, 140))
    faces = [FaceBox(id="a", x=10, y=10, width=40, height=40), FaceBox(id="b", x=500, y=300, width=90, height=60)]

    batch = build_batch(picture, faces, image_size=32, padding=0.18)

    assert batch.shape == (2, 32, 32, 3)
    assert batch.dtype == np.float32
    assert batch.min() >= 0.0 and batch.max() <= 255.0


def test_crop_face_returns_the_padded_region():
    picture = Image.new("RGB", (100, 100), (0, 0, 0))
    picture.paste((255, 255, 255), (40, 40, 60, 60))
    box = FaceBox(id="a", x=40, y=40, width=20, height=20)

    crop = crop_face(picture, box, padding=0.0)

    assert crop.size == (20, 20)
    assert crop.getpixel((10, 10)) == (255, 255, 255)


def test_build_batch_uses_the_configured_padding():
    """A different padding must actually change the crop window."""
    picture = Image.new("RGB", (200, 200), (0, 0, 0))
    picture.paste((255, 255, 255), (80, 80, 120, 120))
    box = FaceBox(id="a", x=80, y=80, width=40, height=40)

    tight = np.asarray(build_batch(picture, [box], 32, 0.0)[0]).mean()
    loose = np.asarray(build_batch(picture, [box], 32, 0.5)[0]).mean()

    assert loose < tight  # more padding -> more black border


def test_sniff_content_type():
    assert sniff_content_type(b"\x89PNG\r\n\x1a\nrest") == "image/png"
    assert sniff_content_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert sniff_content_type(b"GIF89a") is None
    assert sniff_content_type(b"") is None


# ------------------------------------------------------------------ metadata


def test_metadata_padding_prefers_face_padding():
    assert metadata_padding({"face_padding": 0.25}) == 0.25
    assert metadata_padding({"dataset_preparation": {"padding": 0.3}}) == 0.3
    assert metadata_padding({"face_padding": 5}) is None  # out of range is ignored
    assert metadata_padding(None) is None


def test_metadata_labels_and_image_size():
    assert metadata_labels({"labels": ["a", "b"]}) == ("a", "b")
    assert metadata_labels({"labels": "a"}) is None
    assert metadata_labels({}) is None
    assert metadata_image_size({"image_size": 128}) == 128
    assert metadata_image_size({"image_size": True}) is None
    assert metadata_image_size({"image_size": -1}) is None


# ------------------------------------------------------------------ settings


def test_default_model_path_matches_train_py(monkeypatch):
    monkeypatch.setattr("app.config.DEFAULT_MODEL_PATH", __import__("pathlib").Path("outputs/run1/mask_model.keras"))
    settings = Settings.from_env()

    assert settings.model_path.name == "mask_model.keras"
    assert settings.model_path.parent.name == "run1"
    assert settings.metadata_path == settings.model_path.parent / "metadata.json"


def test_settings_read_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MASK_API_MODEL_PATH", str(tmp_path / "custom.keras"))
    monkeypatch.setenv("MASK_API_BATCH_SIZE", "3")
    monkeypatch.setenv("MASK_API_PADDING", "0.25")
    monkeypatch.setenv("MASK_API_MAX_FACES", "5")
    monkeypatch.setenv("MASK_API_REQUIRE_MODEL", "true")
    monkeypatch.setenv("MASK_API_LABELS", "a, b ,c")

    settings = Settings.from_env()

    assert settings.model_path == tmp_path / "custom.keras"
    assert settings.batch_size == 3
    assert settings.padding == 0.25
    assert settings.max_faces == 5
    assert settings.require_model is True
    assert settings.labels == ("a", "b", "c")


def test_relative_model_path_resolves_against_the_project_root(monkeypatch):
    monkeypatch.setenv("MASK_API_MODEL_PATH", "outputs/run2/mask.keras")

    settings = Settings.from_env()

    assert settings.model_path.is_absolute()
    assert settings.model_path.name == "mask.keras"


@pytest.mark.parametrize("name, value", [
    ("MASK_API_BATCH_SIZE", "zero"),
    ("MASK_API_BATCH_SIZE", "0"),
    ("MASK_API_PADDING", "0.9"),
    ("MASK_API_MAX_FACES", "-2"),
    ("MASK_API_REQUIRE_MODEL", "maybe"),
    ("MASK_API_LABELS", " , "),
])
def test_invalid_settings_are_rejected_early(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError):
        Settings.from_env()


def test_canonical_content_type():
    settings = Settings.from_env()

    assert settings.canonical_content_type("image/JPG") == "image/jpeg"
    assert settings.canonical_content_type("image/png; q=1") == "image/png"
    assert settings.canonical_content_type("image/gif") is None
    assert settings.canonical_content_type(None) is None