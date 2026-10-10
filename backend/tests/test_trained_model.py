"""Integration test against the model produced by ``train.py``.

Skipped automatically while ``outputs/run1/mask_model.keras`` is missing, so
``python -m pytest`` stays green before training:

    python train.py && python -m pytest -m realmodel
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT
from app.main import create_app
from conftest import post_predict

MODEL_PATH = PROJECT_ROOT / "outputs" / "run1" / "mask_model.keras"
RAW_IMAGES = PROJECT_ROOT / "data" / "raw" / "images"
RAW_ANNOTATIONS = PROJECT_ROOT / "data" / "raw" / "annotations"
LABELS = {"without_mask", "with_mask", "mask_weared_incorrect"}

pytestmark = [
    pytest.mark.realmodel,
    pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Chua train model (outputs/run1/mask_model.keras)."),
    pytest.mark.skipif(not RAW_IMAGES.is_dir(), reason="Khong co data/raw."),
]


def voc_faces(image_path: Path) -> list[dict]:
    """Pascal VOC boxes converted to the 0-based pixel space of the API."""
    stem = image_path.stem
    matches = sorted(RAW_ANNOTATIONS.glob(f"{stem}.xml")) or sorted(
        xml for xml in RAW_ANNOTATIONS.glob("*.xml") if xml.stem == stem)
    if not matches:
        return []
    root = ET.parse(matches[0]).getroot()
    boxes = []
    for index, obj in enumerate(root.findall("object")):
        box = obj.find("bndbox")
        if box is None:
            continue
        x1 = int(math.floor(float(box.findtext("xmin")))) - 1
        y1 = int(math.floor(float(box.findtext("ymin")))) - 1
        x2 = int(math.ceil(float(box.findtext("xmax"))))
        y2 = int(math.ceil(float(box.findtext("ymax"))))
        if x2 > x1 and y2 > y1:
            boxes.append({"id": f"f{index}", "x": max(0, x1), "y": max(0, y1),
                          "width": x2 - x1, "height": y2 - y1})
    return boxes


def test_trained_model_classifies_real_dataset_faces():
    images = sorted(p for p in RAW_IMAGES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    assert images, "data/raw/images rong"
    checked = 0
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["model"]["ready"] is True
        for image_path in images:
            faces = voc_faces(image_path)
            if not faces:
                continue
            response = post_predict(
                client, faces, image=image_path.read_bytes(),
                filename=image_path.name, content_type="image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png",
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["image"]["width"] > 0 and body["image"]["height"] > 0
            assert len(body["predictions"]) == len(faces)
            assert {p["label"] for p in body["predictions"]} <= LABELS
            assert [p["id"] for p in body["predictions"]] == [f["id"] for f in faces]
            checked += 1
            if checked == 5:
                break
    assert checked > 0, "Khong tim thay anh co nhan de kiem thu."