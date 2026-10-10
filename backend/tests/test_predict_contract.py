"""The published HTTP contract of POST /api/v1/predict and GET /health."""

from __future__ import annotations

import uuid

import pytest

from conftest import face, grid_faces, make_image_bytes, post_predict

RESPONSE_KEYS = {"request_id", "image", "predictions", "total_inference_ms"}
PREDICTION_KEYS = {"id", "label", "confidence", "box", "inference_ms"}
BOX_KEYS = {"x", "y", "width", "height"}
LABELS = {"without_mask", "with_mask", "mask_weared_incorrect"}


def test_single_face_response_matches_contract(client, stub_model):
    response = post_predict(client, [face("face-a", 10, 5, 40, 50)])

    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_KEYS
    assert body["image"] == {"width": 120, "height": 90}
    assert body["request_id"] == response.headers["X-Request-ID"]
    uuid.UUID(body["request_id"])  # parsable request id
    assert isinstance(body["total_inference_ms"], (int, float))

    assert len(body["predictions"]) == 1
    prediction = body["predictions"][0]
    assert set(prediction) == PREDICTION_KEYS
    assert prediction["id"] == "face-a"
    assert prediction["label"] in LABELS
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert set(prediction["box"]) == BOX_KEYS
    assert prediction["box"] == {"x": 10, "y": 5, "width": 40, "height": 50}
    assert all(isinstance(prediction["box"][key], int) for key in BOX_KEYS)
    assert prediction["inference_ms"] >= 0.0
    assert stub_model.calls == [(1, 32, 32, 3)]


def test_predictions_keep_request_order_and_echo_every_box(client):
    faces = [face("a", 0, 0, 30, 30), face("b", 60, 10, 30, 30), face("c", 20, 40, 25, 25)]

    body = post_predict(client, faces).json()

    assert [p["id"] for p in body["predictions"]] == ["a", "b", "c"]
    assert [p["box"] for p in body["predictions"]] == [
        {"x": 0, "y": 0, "width": 30, "height": 30},
        {"x": 60, "y": 10, "width": 30, "height": 30},
        {"x": 20, "y": 40, "width": 25, "height": 25},
    ]
    assert {p["label"] for p in body["predictions"]} <= LABELS


def test_fractional_boxes_are_echoed_unchanged(client):
    body = post_predict(client, [{"id": "f", "x": 1.5, "y": 2.25, "width": 10.5, "height": 9.75}]).json()

    assert body["predictions"][0]["box"] == {"x": 1.5, "y": 2.25, "width": 10.5, "height": 9.75}


def test_confidence_is_the_winning_probability(client):
    body = post_predict(client, [face("only")]).json()

    prediction = body["predictions"][0]
    assert prediction["label"] == "without_mask"
    assert prediction["confidence"] == pytest.approx(0.9, abs=1e-6)


def test_empty_face_list_returns_no_predictions(client, stub_model):
    response = post_predict(client, [])

    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == []
    assert body["total_inference_ms"] == 0.0
    assert body["image"] == {"width": 120, "height": 90}
    assert stub_model.calls == []  # nothing to classify -> no forward pass


def test_faces_are_inferred_in_batches(client_factory, stub_model):
    client = client_factory({"batch_size": 2})

    response = post_predict(client, grid_faces(5))

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 5
    assert stub_model.calls == [(2, 32, 32, 3), (2, 32, 32, 3), (1, 32, 32, 3)]


def test_single_batch_when_under_the_limit(client_factory, stub_model):
    client = client_factory({"batch_size": 16})

    post_predict(client, grid_faces(3))

    assert stub_model.calls == [(3, 32, 32, 3)]


def test_timing_is_reported_per_face_and_totalled(client_factory):
    client = client_factory({"batch_size": 4}, delay=0.04)

    body = post_predict(client, grid_faces(8)).json()

    per_face = [p["inference_ms"] for p in body["predictions"]]
    assert all(value > 5.0 for value in per_face)  # ~40ms shared by 4 faces
    assert body["total_inference_ms"] == pytest.approx(sum(per_face), abs=0.01)
    assert body["total_inference_ms"] >= 60.0  # two batches of 40ms


def test_jpeg_and_png_are_both_accepted(client):
    for image_format, content_type in [("JPEG", "image/jpeg"), ("PNG", "image/png")]:
        response = post_predict(
            client, [face()],
            image=make_image_bytes(image_format), filename=f"shot.{image_format.lower()}",
            content_type=content_type,
        )
        assert response.status_code == 200, (image_format, response.text)
        assert response.json()["image"] == {"width": 120, "height": 90}


def test_crops_are_resized_to_the_model_input(client, stub_model):
    post_predict(client, [face("a", 0, 0, 10, 10), face("b", 100, 80, 20, 10)])

    batch = stub_model.batches[0]
    assert batch.shape == (2, 32, 32, 3)
    assert batch.dtype.name == "float32"
    # Pixels stay in 0..255: normalisation lives inside the exported model.
    assert 0.0 <= batch.min() and batch.max() <= 255.0
    assert batch.std() > 0.0  # a real crop, not a blank canvas


def test_openapi_documents_the_contract(client):
    schema = client.get("/openapi.json").json()

    def resolve(node: dict) -> dict:
        return schema["components"]["schemas"][node["$ref"].rsplit("/", 1)[-1]]

    operation = schema["paths"]["/api/v1/predict"]["post"]
    body = resolve(operation["requestBody"]["content"]["multipart/form-data"]["schema"])
    assert set(body["properties"]) == {"image", "faces"}
    assert set(body["required"]) == {"image", "faces"}
    # OpenAPI 3.1 spells a binary upload as contentMediaType, 3.0 as format=binary.
    image_schema = body["properties"]["image"]
    assert image_schema.get("contentMediaType") == "application/octet-stream" or image_schema.get("format") == "binary"

    response_schema = schema["components"]["schemas"]["PredictResponse"]
    assert set(response_schema["properties"]) == RESPONSE_KEYS
    prediction_schema = resolve(response_schema["properties"]["predictions"]["items"])
    assert set(prediction_schema["properties"]) == PREDICTION_KEYS
    assert resolve(prediction_schema["properties"]["box"]) is schema["components"]["schemas"]["Box"]
    assert set(schema["components"]["schemas"]["Box"]["properties"]) == BOX_KEYS
    assert set(prediction_schema["properties"]["label"]["enum"]) == LABELS

    for code in ["400", "413", "415", "422", "503"]:
        assert code in operation["responses"]


def test_request_id_is_unique_per_request(client):
    first = post_predict(client, [face()]).json()["request_id"]
    second = post_predict(client, [face()]).json()["request_id"]

    assert first != second


def test_missing_model_yields_503_and_health_reports_degraded(client_factory):
    """No weights on disk: the app still starts and explains itself."""
    client = client_factory(ready=False)

    health = client.get("/health").json()
    assert health["status"] == "degraded"
    assert health["ready"] is False
    assert health["model"]["ready"] is False
    assert health["model"]["exists"] is False
    assert "Khong tim thay model" in health["model"]["error"]

    response = post_predict(client, [face()])
    assert response.status_code == 503
    assert "Model chua san sang" in response.json()["detail"]


def test_health_reports_the_loaded_model(client_factory, settings):
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"stub")
    client = client_factory()

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["service"] == "lab3-mask-api"
    assert body["model"] == {
        "ready": True,
        "path": str(settings.model_path),
        "exists": True,
        "labels": ["without_mask", "with_mask", "mask_weared_incorrect"],
        "image_size": 32,
        "padding": 0.18,
        "batch_size": 4,
        "backend": "stub",
        "load_ms": 1.2,
        "loaded_at": "2026-01-01T00:00:00+00:00",
        "last_attempt_at": None,
        "warmup_ms": None,
        "metadata": None,
        "error": None,
    }
    assert body["uptime_seconds"] >= 0.0


def test_health_alias_and_no_store(client):
    assert client.get("/api/v1/health").json()["ready"] is True
    assert client.get("/health").headers["cache-control"] == "no-store"


def test_root_points_at_the_endpoints(client):
    body = client.get("/").json()

    assert body["endpoints"] == {"health": "/health", "predict": "/api/v1/predict", "docs": "/docs"}


def test_model_runtime_failure_is_reported_as_service_unavailable(client, stub_model):
    stub_model.raise_with = RuntimeError("cuda out of memory")

    response = post_predict(client, [face()])

    assert response.status_code == 503
    assert "cuda out of memory" in response.json()["detail"]


def test_logits_are_normalised_to_probabilities(client_factory):
    client = client_factory(logits=True)

    body = post_predict(client, [face()]).json()

    assert body["predictions"][0]["label"] == "without_mask"
    assert body["predictions"][0]["confidence"] == pytest.approx(0.9, abs=1e-4)


def test_label_order_follows_the_model_not_the_request(client_factory):
    """A service configured with a permuted label order keeps its mapping."""
    client = client_factory(labels=("with_mask", "without_mask", "mask_weared_incorrect"))

    body = post_predict(client, [face()]).json()

    assert body["predictions"][0]["label"] == "with_mask"
    assert body["predictions"][0]["confidence"] == pytest.approx(0.9, abs=1e-6)