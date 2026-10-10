"""Input validation: MIME types, decodable images, and face boxes."""

from __future__ import annotations

import json

import pytest

from conftest import IMAGE_PIXELS, face, make_image_bytes, post_predict


def detail(response) -> str:
    body = response.json()
    assert "detail" in body, body
    return body["detail"] if isinstance(body["detail"], str) else json.dumps(body["detail"])


# ----------------------------------------------------------------- MIME types


@pytest.mark.parametrize("content_type", ["image/gif", "image/bmp", "application/pdf", "text/plain"])
def test_unsupported_mime_type_is_415(client, content_type):
    response = post_predict(client, [face()], content_type=content_type)

    assert response.status_code == 415
    assert content_type in detail(response)


def test_image_jpg_alias_is_accepted(client):
    response = post_predict(client, [face()], content_type="image/jpg")

    assert response.status_code == 200


def test_content_type_parameters_are_ignored(client):
    response = post_predict(client, [face()], content_type="image/png; charset=binary")

    assert response.status_code == 200


def test_bytes_are_sniffed_when_the_client_sends_no_type(client):
    response = post_predict(client, [face()], content_type="")

    assert response.status_code == 200
    assert response.json()["image"] == {"width": 120, "height": 90}


def test_mislabelled_upload_uses_the_real_bytes(client, caplog):
    with caplog.at_level("WARNING", logger="lab3.preprocessing"):
        response = post_predict(client, [face()], image=make_image_bytes("PNG"), content_type="image/jpeg")

    assert response.status_code == 200
    assert "khong khop magic bytes" in " ".join(record.getMessage() for record in caplog.records)


def test_non_image_bytes_are_rejected_even_with_an_image_type(client):
    response = post_predict(client, [face()], image=b"this is not an image at all", content_type="image/png")

    assert response.status_code == 400
    assert "anh" in detail(response).lower()


def test_truncated_image_is_rejected(client):
    truncated = make_image_bytes("PNG")[:40]

    response = post_predict(client, [face()], image=truncated, content_type="image/png")

    assert response.status_code == 400


def test_empty_upload_is_rejected(client):
    response = post_predict(client, [face()], image=b"", content_type="image/png")

    assert response.status_code == 400


def test_upload_larger_than_the_byte_cap_is_413(client_factory):
    client = client_factory({"max_image_bytes": 512})

    response = post_predict(client, [face()], image=make_image_bytes("PNG", size=(400, 400)))

    assert response.status_code == 413
    assert "byte" in detail(response)


def test_upload_larger_than_the_pixel_cap_is_413(client_factory):
    client = client_factory({"max_image_pixels": 1000})

    response = post_predict(client, [face()])

    assert response.status_code == 413
    assert "pixel" in detail(response)


# -------------------------------------------------------------- multipart body


def test_missing_image_field_is_422(client):
    response = client.post("/api/v1/predict", data={"faces": json.dumps([face()])})

    assert response.status_code == 422
    assert "image" in detail(response)


def test_missing_faces_field_is_422(client):
    response = client.post(
        "/api/v1/predict",
        files={"image": ("shot.png", make_image_bytes(), "image/png")},
    )

    assert response.status_code == 422
    assert "faces" in detail(response)


def test_blank_faces_field_is_400(client):
    response = post_predict(client, None, raw_faces="   ")

    assert response.status_code == 400
    assert "faces" in detail(response)


# ---------------------------------------------------------------- faces payload


@pytest.mark.parametrize("payload", ["{", "not json", "[{'id': 'f1'}]", '{"id": "f1"}'])
def test_invalid_faces_payload_is_400(client, payload):
    response = post_predict(client, None, raw_faces=payload)

    assert response.status_code == 400
    assert "faces" in detail(response) or "JSON" in detail(response)


def test_faces_must_be_an_array(client):
    response = post_predict(client, {"id": "f1", "x": 0, "y": 0, "width": 10, "height": 10})

    assert response.status_code == 400
    assert "array" in detail(response)


@pytest.mark.parametrize("missing", ["id", "x", "y", "width", "height"])
def test_face_missing_a_field_is_400(client, missing):
    payload = face("f1")
    payload.pop(missing)

    response = post_predict(client, [payload])

    assert response.status_code == 400
    assert missing in detail(response)


@pytest.mark.parametrize("value", [None, "", "   ", 7, [], {}])
def test_face_with_a_bad_id_is_400(client, value):
    payload = face("f1")
    payload["id"] = value

    response = post_predict(client, [payload])

    assert response.status_code == 400
    assert "id" in detail(response)


@pytest.mark.parametrize("value", [-1, -0.5])
def test_negative_origin_is_400(client, value):
    response = post_predict(client, [face("f1", value, 0, 10, 10)])

    assert response.status_code == 400
    assert "am" in detail(response)


@pytest.mark.parametrize("value", [0, -10])
def test_non_positive_size_is_400(client, value):
    response = post_predict(client, [face("f1", 0, 0, value, 10)])

    assert response.status_code == 400
    assert "lon hon 0" in detail(response)


@pytest.mark.parametrize("value", ["abc", None, [1], {}])
def test_non_numeric_coordinates_are_400(client, value):
    payload = face("f1")
    payload["x"] = value

    response = post_predict(client, [payload])

    assert response.status_code == 400


def test_booleans_are_not_coordinates(client):
    payload = face("f1")
    payload["width"] = True

    response = post_predict(client, [payload])

    assert response.status_code == 400


@pytest.mark.parametrize("box", [
    {"x": 100, "y": 0, "width": 40, "height": 40},    # right edge 140 > 120
    {"x": 0, "y": 80, "width": 40, "height": 40},     # bottom edge 120 > 90
    {"x": 0, "y": 0, "width": 200, "height": 40},     # wider than the image
])
def test_box_outside_the_image_is_400(client, box):
    response = post_predict(client, [{"id": "f1", **box}])

    assert response.status_code == 400
    assert "vuot khung anh" in detail(response)


def test_error_names_the_offending_face_index_and_id(client):
    faces = [face("ok", 0, 0, 10, 10), face("bad", 100, 0, 40, 40)]

    response = post_predict(client, faces)

    assert response.status_code == 400
    assert "faces[1]" in detail(response)
    assert "bad" in detail(response)


def test_too_many_faces_is_400(client_factory, stub_model):
    client = client_factory({"max_faces": 2})

    response = post_predict(client, [face(f"f{i}", i * 20, 0, 10, 10) for i in range(3)])

    assert response.status_code == 400
    assert "3" in detail(response)
    assert stub_model.calls == []


def test_oversized_faces_field_is_413(client_factory):
    """Starlette caps *field* parts at 1MB, so only a unit test can reach this."""
    from app.errors import ApiError
    from app.preprocessing import parse_faces

    settings = client_factory().app.state.settings
    with pytest.raises(ApiError) as error:
        parse_faces("[" + " " * 2_000_001 + "]", IMAGE_PIXELS, settings)
    assert error.value.status_code == 413


def test_numeric_strings_are_accepted(client):
    response = post_predict(client, [{"id": "f1", "x": "5", "y": "5", "width": "10", "height": "10.5"}])

    assert response.status_code == 200
    assert response.json()["predictions"][0]["box"] == {"x": 5, "y": 5, "width": 10, "height": 10.5}


def test_unknown_extra_keys_are_ignored(client):
    response = post_predict(client, [dict(face("f1"), score=0.98, class_id=7)])

    assert response.status_code == 200


def test_integer_and_float_coordinates_stay_numeric(client):
    body = post_predict(client, [face("f1", 0, 0, 10, 10)]).json()

    box = body["predictions"][0]["box"]
    assert all(isinstance(box[key], (int, float)) for key in box)
    assert IMAGE_PIXELS == (120, 90) == (body["image"]["width"], body["image"]["height"])


def test_validation_runs_before_the_model_is_used(client, stub_model):
    response = post_predict(client, [face("f1", 500, 500, 10, 10)])

    assert response.status_code == 400
    assert stub_model.calls == []


def test_image_size_is_reported_after_exif_transpose(client):
    """Coordinates are interpreted in the orientation a viewer sees."""
    import io

    from PIL import Image

    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise
    buffer = io.BytesIO()
    Image.new("RGB", (40, 90), (10, 20, 30)).save(buffer, format="JPEG", exif=exif)

    response = post_predict(client, [face("f1", 0, 0, 40, 40)],
                            image=buffer.getvalue(), filename="r.jpg", content_type="image/jpeg")

    assert response.status_code == 200
    assert response.json()["image"] == {"width": 90, "height": 40}