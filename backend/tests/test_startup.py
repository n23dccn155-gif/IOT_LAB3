"""Startup behaviour: readiness logging, degraded serving and fail-fast."""

from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

from app.main import create_app, get_model_service
from app.model_service import ModelService
from conftest import face, post_predict, write_metadata, write_weights
from test_model_service import FakeKeras, install_fake_keras


def test_startup_survives_a_missing_model(settings):
    app = create_app(settings)

    with TestClient(app) as client:  # must not raise
        assert client.get("/health").status_code == 200
        assert client.get("/health").json()["status"] == "degraded"


def test_startup_logs_the_resolved_model_path(settings, caplog):
    with caplog.at_level("INFO", logger="lab3.api"):
        with TestClient(create_app(settings)):
            pass

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "Khoi dong lab3-mask-api" in messages
    assert str(settings.model_path) in messages
    assert "Model CHUA san sang" in messages


def test_require_model_aborts_startup(settings):
    settings = dataclasses.replace(settings, require_model=True)

    with pytest.raises(Exception) as error:
        with TestClient(create_app(settings)):
            pass

    assert "ModelNotReadyError" in type(error.value).__name__


def test_load_at_startup_false_defers_loading(monkeypatch, settings, stub_model):
    fake = install_fake_keras(monkeypatch, FakeKeras(model=stub_model))
    write_weights(settings)
    active = dataclasses.replace(settings, load_at_startup=False)
    service = ModelService(active)
    app = create_app(active, service)

    with TestClient(app) as client:
        assert fake.calls == []
        assert client.get("/health").json()["model"]["ready"] is False

        response = post_predict(client, [face("f1", 0, 0, 40, 40)])

    assert response.status_code == 200
    assert response.json()["predictions"][0]["label"] == "without_mask"
    assert client.get("/health").json()["status"] == "ok"


def test_model_is_loaded_once_at_startup(monkeypatch, settings, stub_model):
    fake = install_fake_keras(monkeypatch, FakeKeras(model=stub_model))
    write_weights(settings)
    app = create_app(settings, ModelService(settings))

    with TestClient(app) as client:
        post_predict(client, [face()])
        post_predict(client, [face()])

    assert len(fake.calls) == 1


def test_model_is_reloadable(monkeypatch, settings, stub_model):
    install_fake_keras(monkeypatch, FakeKeras(model=stub_model))
    write_weights(settings)
    service = ModelService(settings)

    first = service.load()
    second = service.load(force=True)

    assert first is not second
    assert service.ready is True


def test_dependency_override_seam_is_used(monkeypatch, settings, stub_model):
    """Tests can swap the service without touching the app factory."""
    install_fake_keras(monkeypatch, FakeKeras(model=stub_model))
    write_weights(settings)
    service = ModelService(settings)
    service.load()
    app = create_app(settings, ModelService(settings))
    app.dependency_overrides[get_model_service] = lambda: service

    with TestClient(app) as client:
        assert post_predict(client, [face()]).status_code == 200
    assert stub_model.calls == [(1, 32, 32, 3)]


def test_health_reports_metadata_when_present(monkeypatch, settings, stub_model):
    install_fake_keras(monkeypatch, FakeKeras(model=stub_model))
    write_weights(settings)
    write_metadata(settings.metadata_path, architecture="MobileNetV2 alpha=0.35")
    app = create_app(settings, ModelService(settings))

    with TestClient(app) as client:
        model = client.get("/health").json()["model"]

    assert model["ready"] is True
    assert model["metadata"]["architecture"] == "MobileNetV2 alpha=0.35"
    assert model["exists"] is True