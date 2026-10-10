"""Uniform ``{"detail": ...}`` errors for the LAB3 predict API."""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger("lab3.errors")

DETAIL_KEY = "detail"


class ApiError(HTTPException):
    """An HTTP error whose body is exactly ``{"detail": <message>}``."""

    def __init__(self, status_code: int, detail: str, headers: dict | None = None) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class ModelNotReadyError(RuntimeError):
    """Raised when inference is requested but no usable model is loaded."""


def bad_request(detail: str) -> ApiError:
    return ApiError(status_code=400, detail=detail)


def unsupported_media_type(detail: str) -> ApiError:
    return ApiError(status_code=415, detail=detail)


def payload_too_large(detail: str) -> ApiError:
    return ApiError(status_code=413, detail=detail)


def service_unavailable(detail: str) -> ApiError:
    return ApiError(status_code=503, detail=detail, headers={"Retry-After": "30"})


def _flatten(errors: list[dict]) -> str:
    parts = []
    for error in errors:
        location = ".".join(str(item) for item in error.get("loc", ()) if item != "body")
        message = error.get("msg", "gia tri khong hop le")
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) or "Du lieu gui len khong hop le."


async def request_validation_handler(_: Request, error: Exception) -> JSONResponse:
    """Keep FastAPI's 422 body but expose it as a readable ``detail`` string."""
    errors = error.errors() if isinstance(error, RequestValidationError) else []
    return JSONResponse(status_code=422, content={DETAIL_KEY: _flatten(errors)})


async def unhandled_handler(request: Request, error: Exception) -> JSONResponse:
    LOGGER.exception("Loi khong mong doi khi xu ly %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={DETAIL_KEY: "Loi may chu: hay thu lai sau."})