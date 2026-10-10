"""Image decoding, face validation and cropping for the LAB3 predict API.

The crop maths is a deliberate copy of ``prepare_dataset.prepare`` so that a
face served by the API is framed exactly like the crops the model was trained
on:

    side  = min(max(w, h) * (1 + 2 * padding), image_width, image_height)
    left  = clamp((x + w - side) / 2, 0, image_width - side)
    top   = clamp((y + h - side) / 2, 0, image_height - side)

Preprocessing mirrors ``common.image_array``: EXIF transpose, RGB, square
resize, ``float32`` in the ``0..255`` range. Normalisation lives *inside* the
exported Keras model, so nothing is normalised here.
"""

from __future__ import annotations

import io
import json
import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .errors import ApiError, bad_request, payload_too_large
from .schemas import FaceRequest

LOGGER = logging.getLogger("lab3.preprocessing")

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"

ResizeMethod = Image.Resampling.BILINEAR


@dataclass(frozen=True)
class FaceBox:
    """A validated detector box in pixels of the decoded image."""

    id: str
    x: float
    y: float
    width: float
    height: float

    def as_payload(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


def sniff_content_type(data: bytes) -> str | None:
    """Identify JPEG/PNG from the magic bytes, ignoring client headers."""
    if data.startswith(PNG_SIGNATURE):
        return "image/png"
    if data.startswith(JPEG_SIGNATURE):
        return "image/jpeg"
    return None


def check_content_type(declared: str | None, data: bytes, settings: Settings) -> str:
    """Reject unsupported uploads; return the decoder key to use.

    The declared MIME type is validated strictly (415) because it is part of
    the published contract, but the actual bytes win when the two disagree:
    Pillow decodes from content anyway and a mislabelled JPEG is still a JPEG.
    """
    allowed = settings.content_types
    canonical = settings.canonical_content_type(declared)
    if declared and canonical is None:
        raise ApiError(
            status_code=415,
            detail=f"Content-Type {declared!r} khong duoc ho tro. Chi chap nhan: {', '.join(allowed)}.",
        )
    sniffed = sniff_content_type(data)
    if sniffed is None or sniffed not in allowed:
        found = sniffed or "khong ro"
        raise bad_request(f"Anh phai la JPEG hoac PNG (magic bytes: {found}).")
    if canonical and canonical != sniffed:
        LOGGER.warning("Content-Type %s khong khop magic bytes %s; dung bytes.", canonical, sniffed)
    return sniffed


def decode_image(data: bytes, settings: Settings) -> Image.Image:
    """Decode an upload into an RGB image, EXIF orientation applied."""
    if not data:
        raise bad_request("File anh rong.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise bad_request(f"Khong doc duoc file anh: {error}.") from error
    if width * height > settings.max_image_pixels:
        raise payload_too_large(
            f"Anh co {width * height} pixel, vuot gioi han {settings.max_image_pixels} pixel."
        )
    try:
        with Image.open(io.BytesIO(data)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise bad_request(f"Khong doc duoc file anh: {error}.") from error


def parse_faces(payload: str, image_size: tuple[int, int], settings: Settings) -> list[FaceBox]:
    """Parse and validate the ``faces`` JSON array form field."""
    if payload is None or not payload.strip():
        raise bad_request("Thieu truong 'faces' (chuoi JSON array).")
    if len(payload) > 2_000_000:
        raise payload_too_large("Truong 'faces' qua lon.")
    try:
        document = json.loads(payload)
    except ValueError as error:
        raise bad_request(f"'faces' khong phai JSON hop le: {error}.") from error
    if not isinstance(document, list):
        raise bad_request("'faces' phai la JSON array, ví du [{\"id\":\"f1\",\"x\":0,...}].")
    if len(document) > settings.max_faces:
        raise bad_request(f"Cho toi da {settings.max_faces} moi request, nhan {len(document)}.")
    width, height = image_size
    faces: list[FaceBox] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(document):
        face = _parse_face(item, index, width, height)
        if face.id in seen_ids:
            raise bad_request(f"faces[{index}] co id bi trung {face.id!r}; moi khuon mat can id rieng.")
        seen_ids.add(face.id)
        faces.append(face)
    return faces


def _parse_face(item: object, index: int, image_width: int, image_height: int) -> FaceBox:
    if not isinstance(item, dict):
        raise bad_request(f"faces[{index}] phai la JSON object.")
    try:
        face = FaceRequest.model_validate(item)
    except Exception as error:  # pydantic ValidationError and friends
        raise bad_request(f"faces[{index}] khong hop le: {_describe(error)}.") from error
    missing = [key for key in ("id", "x", "y", "width", "height") if key not in item]
    if missing:
        raise bad_request(f"faces[{index}] thieu truong: {', '.join(missing)}.")
    right = face.x + face.width
    bottom = face.y + face.height
    if right > image_width or bottom > image_height:
        raise bad_request(
            f"faces[{index}] (id={face.id!r}) vuot khung anh "
            f"[0,0,{image_width},{image_height}]: can {right}x{bottom} pixel."
        )
    return FaceBox(id=face.id, x=float(face.x), y=float(face.y),
                   width=float(face.width), height=float(face.height))


def _describe(error: Exception) -> str:
    errors = getattr(error, "errors", None)
    if callable(errors):
        messages = []
        for item in errors():
            location = ".".join(str(part) for part in item.get("loc", ()) if part != "faces")
            messages.append(f"{location}: {item.get('msg')}" if location else item.get("msg", ""))
        return "; ".join(messages)
    return str(error)


def face_crop_bounds(box: FaceBox, image_size: tuple[int, int], padding: float) -> tuple[int, int, int, int]:
    """Square crop bounds for ``box``; mirrors ``prepare_dataset.prepare``."""
    width, height = image_size
    side = min(max(box.width, box.height) * (1 + 2 * padding), width, height)
    side = max(side, 1.0)
    # ``(x1 + x2 - side) / 2`` with x2 = x1 + width: centre the square on the box.
    left = max(0.0, min(width - side, box.x + box.width / 2 - side / 2))
    top = max(0.0, min(height - side, box.y + box.height / 2 - side / 2))
    return (
        int(left),
        int(top),
        min(width, math.ceil(left + side)),
        min(height, math.ceil(top + side)),
    )


def crop_face(image: Image.Image, box: FaceBox, padding: float) -> Image.Image:
    """Return the padded, square RGB crop for one face."""
    return image.crop(face_crop_bounds(box, image.size, padding))


def build_batch(
    image: Image.Image,
    faces: Sequence[FaceBox],
    image_size: int,
    padding: float,
) -> np.ndarray:
    """Stack crops into ``(n, image_size, image_size, 3)`` float32 in 0..255."""
    batch = np.empty((len(faces), image_size, image_size, 3), dtype=np.float32)
    for index, face in enumerate(faces):
        crop = ImageOps.fit(crop_face(image, face, padding), (image_size, image_size), ResizeMethod)
        batch[index] = np.asarray(crop, dtype=np.float32)
    return batch
