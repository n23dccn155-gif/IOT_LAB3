"""Model loading, readiness transitions and inference helpers."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from app.config import Settings, read_metadata
from app.errors import ModelNotReadyError
from app.model_service import ModelService, _probabilities
from app.preprocessing import FaceBox
from conftest import StubModel, make_image_bytes, make_loaded, write_metadata, write_weights
from PIL import Image


class FakeKeras:
    """Stands in for the ``keras`` module so the loader can be exercised fast."""

    def __init__(self, model=None, error: Exception | None = None):
        self.model = model
        self.error = error
        self.calls: list[tuple] = []

        class Backend:
            @staticmethod
            def backend() -> str:
                return "fake-keras"

        class Models:
            @staticmethod
            def load_model(path, compile=False):
                self_backend.calls.append((path, compile))
                if self_backend.error is not None:
                    raise self_backend.error
                return self_backend.model

        self_backend = self
        self.backend = Backend()
        self.models = Models()


def install_fake_keras(monkeypatch, fake):
    monkeypatch.setattr("app.model_service.import_keras", lambda: fake)
    return fake


# ------------------------------------------------------------------- loading


def test_missing_weights_raise_and_are_reported(settings):
    service = ModelService(settings)

    with pytest.raises(ModelNotReadyError) as error:
        service.load()

    assert "Khong tim thay model" in str(error.value)
    status = service.status()
    assert status["ready"] is False
    assert status["exists"] is False
    assert "Khong tim thay model" in status["error"]


def test_failed_load_is_retried_only_after_the_backoff(monkeypatch, settings):
    write_weights(settings)
    fake = install_fake_keras(monkeypatch, FakeKeras(error=OSError("corrupt file")))
    service = ModelService(settings)

    for _ in range(2):
        with pytest.raises(ModelNotReadyError):
            service.ensure_ready()

    assert len(fake.calls) == 1, "a failed load must not hammer Keras on every request"
    assert service.ready is False


def test_backoff_expired_loads_again(monkeypatch, settings):
    write_weights(settings)
    fake = install_fake_keras(monkeypatch, FakeKeras(error=OSError("corrupt file")))
    service = ModelService(dataclasses.replace(settings, retry_seconds=0.0))

    for _ in range(3):
        with pytest.raises(ModelNotReadyError):
            service.ensure_ready()

    assert len(fake.calls) == 3


def test_lazy_load_happens_on_the_first_request(settings, monkeypatch):
    stub = StubModel()
    install_fake_keras(monkeypatch, FakeKeras(model=stub))
    write_weights(settings)
    service = ModelService(dataclasses.replace(settings, load_at_startup=False))

    assert service.ready is False

    service.load()

    assert service.ready is True
    assert service.status()["backend"] == "fake-keras"


def test_metadata_drives_labels_image_size_and_padding(monkeypatch, settings):
    stub = StubModel(image_size=64)
    install_fake_keras(monkeypatch, FakeKeras(model=stub))
    write_metadata(settings.metadata_path, image_size=64, face_padding=0.12,
                   labels=["without_mask", "with_mask", "mask_weared_incorrect"])
    write_weights(settings)

    loaded = ModelService(settings).load()

    assert loaded.labels == ("without_mask", "with_mask", "mask_weared_incorrect")
    assert loaded.image_size == 64
    assert loaded.padding == 0.12
    assert loaded.metadata["image_size"] == 64


def test_padding_falls_back_to_the_setting_without_metadata(monkeypatch, settings):
    install_fake_keras(monkeypatch, FakeKeras(model=StubModel()))
    write_weights(settings)

    loaded = ModelService(dataclasses.replace(settings, padding=0.3)).load()

    assert loaded.padding == 0.3
    assert loaded.metadata is None


def test_metadata_padding_nested_in_dataset_preparation(monkeypatch, settings):
    install_fake_keras(monkeypatch, FakeKeras(model=StubModel()))
    write_metadata(settings.metadata_path, face_padding=None, dataset_preparation={"padding": 0.42})
    write_weights(settings)

    assert ModelService(settings).load().padding == 0.42


def test_model_with_the_wrong_number_of_outputs_is_rejected(monkeypatch, settings):
    install_fake_keras(monkeypatch, FakeKeras(model=StubModel(units=2)))
    write_weights(settings)

    with pytest.raises(ModelNotReadyError) as error:
        ModelService(settings).load()

    assert "so nhan" in str(error.value)


def test_model_with_a_non_square_input_is_rejected(monkeypatch, settings):
    stub = StubModel()
    stub.input_shape = (None, 64, 96, 3)
    install_fake_keras(monkeypatch, FakeKeras(model=stub))
    write_weights(settings)

    with pytest.raises(ModelNotReadyError) as error:
        ModelService(settings).load()

    assert "vuong" in str(error.value)


def test_unknown_labels_are_rejected(monkeypatch, settings):
    install_fake_keras(monkeypatch, FakeKeras(model=StubModel()))
    write_weights(settings)

    with pytest.raises(ModelNotReadyError) as error:
        ModelService(dataclasses.replace(settings, labels=("a", "b", "c"))).load()

    assert "khong hop le" in str(error.value)


def test_warmup_runs_one_forward_pass(monkeypatch, settings):
    stub = StubModel()
    install_fake_keras(monkeypatch, FakeKeras(model=stub))
    write_weights(settings)

    loaded = ModelService(dataclasses.replace(settings, warmup=True)).load()

    assert stub.calls == [(1, 32, 32, 3)]
    assert loaded.warmup_ms >= 0.0


def test_load_is_idempotent(monkeypatch, settings):
    fake = install_fake_keras(monkeypatch, FakeKeras(model=StubModel()))
    write_weights(settings)
    service = ModelService(settings)

    service.load()
    service.load()

    assert len(fake.calls) == 1
    assert service.ready is True


def test_corrupt_metadata_is_ignored(monkeypatch, settings):
    install_fake_keras(monkeypatch, FakeKeras(model=StubModel()))
    settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metadata_path.write_text("{not json", encoding="utf-8")
    write_weights(settings)

    assert ModelService(settings).load().metadata is None


# ----------------------------------------------------------------- inference


def test_classify_returns_one_outcome_per_face(service, stub_model):
    picture = Image.open(__import__("io").BytesIO(make_image_bytes())).convert("RGB")
    faces = [FaceBox(id=f"f{i}", x=i * 20, y=0, width=20, height=20) for i in range(3)]

    outcomes, total_ms = service.classify(picture, faces)

    assert [o.index for o in outcomes] == [0, 1, 2]
    assert {o.label for o in outcomes} == {"without_mask"}
    assert all(o.confidence == pytest.approx(0.9, abs=1e-6) for o in outcomes)
    assert all(len(o.probabilities) == 3 for o in outcomes)
    assert total_ms == pytest.approx(sum(o.inference_ms for o in outcomes), abs=0.01)
    assert stub_model.calls == [(3, 32, 32, 3)]


def test_classify_without_faces_does_not_call_the_model(service, stub_model):
    outcomes, total_ms = service.classify(Image.new("RGB", (32, 32)), [])

    assert outcomes == []
    assert total_ms == 0.0
    assert stub_model.calls == []


def test_classify_raises_when_the_model_is_missing(settings):
    picture = Image.new("RGB", (32, 32))
    service = ModelService(settings)

    with pytest.raises(ModelNotReadyError):
        service.classify(picture, [FaceBox(id="f", x=0, y=0, width=8, height=8)])


def test_classify_rejects_unexpected_output_width(settings):
    service = ModelService(settings, loaded=make_loaded(StubModel(units=2), settings))
    picture = Image.new("RGB", (32, 32))
    faces = [FaceBox(id="f", x=0, y=0, width=8, height=8)]

    with pytest.raises(RuntimeError, match="lop"):
        service.classify(picture, faces)


# ------------------------------------------------------------- probabilities


def test_probabilities_passthrough_for_softmax_output():
    values = np.array([[0.2, 0.5, 0.3], [1.0, 0.0, 0.0]])

    assert np.allclose(_probabilities(None, values, 3), values, atol=1e-6)


def test_probabilities_apply_softmax_to_logits():
    logits = np.array([[2.0, 1.0, 0.1]])

    result = _probabilities(None, logits, 3)

    assert result.shape == (1, 3)
    assert result.sum() == pytest.approx(1.0)
    assert result[0, 0] > result[0, 1] > result[0, 2]


def test_probabilities_accept_a_single_vector():
    assert _probabilities(None, np.array([0.1, 0.2, 0.7]), 3).shape == (1, 3)


@pytest.mark.parametrize("bad", [np.array([np.nan, 0.5, 0.5]), np.zeros((2, 2, 2))])
def test_probabilities_reject_broken_output(bad):
    with pytest.raises(RuntimeError):
        _probabilities(None, bad, 3)


def test_probabilities_reject_a_wrong_number_of_classes():
    with pytest.raises(RuntimeError):
        _probabilities(None, np.array([[0.5, 0.5]]), 3)


# ----------------------------------------------------------------- read_metadata


def test_read_metadata_handles_missing_and_broken_files(tmp_path):
    assert read_metadata(tmp_path / "nope.json") is None
    assert read_metadata(None) is None
    broken = tmp_path / "broken.json"
    broken.write_text("[1, 2]", encoding="utf-8")
    assert read_metadata(broken) is None
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"labels": ["a", "b", "c"]}), encoding="utf-8")
    assert read_metadata(good) == {"labels": ["a", "b", "c"]}


def test_settings_public_dict_has_no_secrets(settings):
    public = settings.as_public_dict()

    assert set(public) >= {"model_path", "labels", "image_size", "padding", "batch_size", "max_faces"}
    assert isinstance(Settings.from_env().as_public_dict(), dict)