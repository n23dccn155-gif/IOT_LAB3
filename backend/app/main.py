"""FastAPI application for the LAB3 mask-classification API.

    uvicorn app.main:app --app-dir backend
    python -m app                       # same, with argparse

Endpoints
    GET  /health            readiness of the service and the Keras model
    POST /api/v1/predict     classify every face box of an uploaded image

Importing this module never touches Keras, so the app (and the test suite)
imports cleanly before the model has been trained.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator, Sequence

from fastapi import Depends, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool

from . import __version__
from .config import Settings
from .errors import (
    ApiError,
    ModelNotReadyError,
    payload_too_large,
    request_validation_handler,
    service_unavailable,
    unhandled_handler,
)
from .model_service import FaceOutcome, ModelService
from .preprocessing import FaceBox, check_content_type, decode_image, parse_faces
from .schemas import (
    Box,
    ErrorResponse,
    HealthResponse,
    ImageSize,
    PredictResponse,
    Prediction,
    as_json_number,
)

LOGGER = logging.getLogger("lab3.api")

SERVICE_NAME = "lab3-mask-api"
PREDICT_PATH = "/api/v1/predict"
STARTED_AT = time.monotonic()

PREDICT_DESCRIPTION = """
Classify every supplied face box with the Keras model exported by `train.py`.

`image` must be JPEG or PNG and `faces` a JSON array of
`{"id": str, "x": float, "y": float, "width": float, "height": float}` boxes in
pixels of the uploaded image. Boxes are cropped with the training padding
(`metadata.json: face_padding`) before a batched forward pass.
""".strip()

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid image, JSON or face box"},
    413: {"model": ErrorResponse, "description": "Image or face list too large"},
    415: {"model": ErrorResponse, "description": "Unsupported image MIME type"},
    422: {"model": ErrorResponse, "description": "Malformed multipart body"},
    503: {"model": ErrorResponse, "description": "Model not loaded yet"},
}


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )


def get_model_service(request: Request) -> ModelService:
    """Dependency seam: tests override this with a preloaded service."""
    return request.app.state.model_service


def build_predict_response(
    request_id: str,
    image_size: tuple[int, int],
    outcomes: Sequence[FaceOutcome],
    faces: Sequence[FaceBox],
    total_ms: float,
) -> PredictResponse:
    predictions = []
    for face, outcome in zip(faces, outcomes):
        box = face.as_payload()
        predictions.append(Prediction(
            id=face.id,
            label=outcome.label,
            confidence=outcome.confidence,
            box=Box(**{key: as_json_number(value) for key, value in box.items()}),
            inference_ms=outcome.inference_ms,
        ))
    return PredictResponse(
        request_id=request_id,
        image=ImageSize(width=image_size[0], height=image_size[1]),
        predictions=predictions,
        total_inference_ms=total_ms,
    )


def create_app(settings: Settings | None = None, service: ModelService | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    model_service = service or ModelService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        path = settings.model_path
        LOGGER.info(
            "Khoi dong %s v%s | model=%s | ton tai=%s | load_at_startup=%s | batch=%s | max_faces=%s",
            SERVICE_NAME, __version__, path, path.is_file(), settings.load_at_startup,
            settings.batch_size, settings.max_faces,
        )
        if settings.load_at_startup:
            try:
                app.state.model_service.load()
            except ModelNotReadyError as error:
                if settings.require_model:
                    LOGGER.error("MASK_API_REQUIRE_MODEL=true: dung khoi dong.")
                    raise
                LOGGER.warning(
                    "Model CHUA san sang (ready=false): %s | POST %s tra 503 cho den khi nap duoc.",
                    error, PREDICT_PATH,
                )
        else:
            LOGGER.info("MASK_API_LOAD_AT_STARTUP=false: model se nap o request dau tien.")
        yield
        LOGGER.info("Dung %s.", SERVICE_NAME)

    app = FastAPI(
        title="LAB3 Mask Classification API",
        version=__version__,
        summary="Phan loai deo khau trang tren tung khuon mat.",
        description=(
            "Classify cropped faces with the MobileNetV2 model trained in LAB3. "
            "The model path is configurable with `MASK_API_MODEL_PATH` (default "
            f"`{settings.model_path}`); readiness is reported by `GET /health`."
        ),
        lifespan=lifespan,
        responses={500: {"model": ErrorResponse, "description": "Unexpected server error"}},
    )
    app.state.settings = settings
    app.state.model_service = model_service
    app.add_exception_handler(RequestValidationError, request_validation_handler)
    app.add_exception_handler(Exception, unhandled_handler)

    @app.get("/", include_in_schema=False)
    async def index() -> dict:
        return {
            "service": SERVICE_NAME,
            "version": __version__,
            "endpoints": {"health": "/health", "predict": PREDICT_PATH, "docs": "/docs"},
            "model": {"ready": model_service.ready, "path": str(settings.model_path)},
        }

    @app.get("/health", response_model=HealthResponse, tags=["ops"], summary="Service and model readiness")
    @app.get("/api/v1/health", response_model=HealthResponse, include_in_schema=False)
    async def health(response: Response) -> HealthResponse:
        status = model_service.status()
        response.headers["Cache-Control"] = "no-store"
        return HealthResponse(
            status="ok" if status["ready"] else "degraded",
            ready=status["ready"],
            service=SERVICE_NAME,
            version=__version__,
            uptime_seconds=round(time.monotonic() - STARTED_AT, 3),
            model=status,
        )

    @app.post(
        PREDICT_PATH,
        response_model=PredictResponse,
        tags=["inference"],
        summary="Classify face boxes of an uploaded image",
        description=PREDICT_DESCRIPTION,
        responses=ERROR_RESPONSES,
    )
    async def predict(
        request: Request,
        response: Response,
        image: Annotated[UploadFile, File(description="Anh JPEG/PNG chua nhieu khuon mat.")],
        faces: Annotated[str, Form(description='JSON array: [{"id":"f1","x":0,"y":0,"width":64,"height":64}]')],
        service: Annotated[ModelService, Depends(get_model_service)],
    ) -> PredictResponse:
        started = time.perf_counter()
        data = await image.read(settings.max_image_bytes + 1)
        if len(data) > settings.max_image_bytes:
            raise payload_too_large(f"Anh vuot gioi han {settings.max_image_bytes} byte.")
        check_content_type(image.content_type, data, settings)
        picture = await run_in_threadpool(decode_image, data, settings)
        boxes = parse_faces(faces, picture.size, settings)

        try:
            outcomes, total_ms = await run_in_threadpool(service.classify, picture, boxes)
        except ModelNotReadyError as error:
            raise service_unavailable(f"Model chua san sang: {error}") from error
        except ApiError:
            raise
        except Exception as error:  # noqa: BLE001 - any Keras runtime failure
            LOGGER.exception("Inference that bai.")
            raise service_unavailable(f"Khong chay duoc model: {error}.") from error

        request_id = str(uuid.uuid4())
        response.headers["X-Request-ID"] = request_id
        LOGGER.debug(
            "%s %s | faces=%d | inference=%.1fms | total=%.1fms",
            request.method, PREDICT_PATH, len(boxes), total_ms,
            (time.perf_counter() - started) * 1000.0,
        )
        return build_predict_response(request_id, picture.size, outcomes, boxes, total_ms)

    return app


app = create_app()