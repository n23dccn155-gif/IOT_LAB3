"""``python -m app`` entry point with plain argparse instead of uvicorn flags."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app", description="Chay LAB3 mask API.")
    parser.add_argument("--host", default=os.environ.get("MASK_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MASK_API_PORT", "8000")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MASK_API_WORKERS", "1")))
    parser.add_argument("--log-level", default=os.environ.get("MASK_API_LOG_LEVEL", "info").lower())
    args = parser.parse_args()
    import uvicorn

    if args.workers > 1:
        uvicorn.run("app.main:app", host=args.host, port=args.port,
                    workers=args.workers, log_level=args.log_level)
        return
    from .main import app

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()