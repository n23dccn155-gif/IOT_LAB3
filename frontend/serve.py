"""Máy chủ tĩnh cho giao diện LAB3, kèm proxy tùy chọn tới API FastAPI.

Chạy:
    python serve.py                                  # phục vụ file + proxy tới 127.0.0.1:8000
    python serve.py --port 5500
    python serve.py --api http://127.0.0.1:8000 --host 0.0.0.0
    python serve.py --no-proxy                       # chỉ phục vụ file tĩnh

Vì sao cần máy chủ: trình duyệt chỉ cho phép getUserMedia khi trang chạy trên
http://localhost hoặc https, nên mở file:// thì không mở được camera.

Nếu bật proxy, các đường dẫn /api/... và /health được chuyển tiếp tới FastAPI.
Khi đó trang và API cùng origin nên không gặp vấn đề CORS: để ô "Địa chỉ API"
trong giao diện là trống hoặc đúng địa chỉ đang mở. Nếu tắt proxy, cần nhập địa
chỉ API đầy đủ và bảo đảm FastAPI đã bật CORSMiddleware cho origin của trang.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import socketserver
import sys
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PORT = 5500
DEFAULT_HOST = "127.0.0.1"
DEFAULT_API = "http://127.0.0.1:8000"
PROXY_PATHS = ("/api/", "/health", "/docs", "/redoc", "/openapi.json")
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-encoding",
    "content-length",
    "host",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    api_base = ""
    server_version = "Lab3Frontend/1.0"

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml",
        ".wasm": "application/wasm",
    }

    def guess_type(self, path: str) -> str:
        kind = super().guess_type(path)
        if kind in ("text/html", "text/css", "text/javascript", "application/json"):
            return kind + "; charset=utf-8"
        return kind

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _proxy_request(self) -> bool:
        path = self.path.split("?", 1)[0]
        return bool(self.api_base) and any(
            path == prefix.rstrip("/") or path.startswith(prefix) for prefix in PROXY_PATHS
        )

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {key: value for key, value in self.headers.items() if key.lower() not in HOP_BY_HOP}
        request = urllib.request.Request(
            self.api_base + self.path, data=body, method=self.command, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                self._forward(response.status, dict(response.headers), response.read())
        except urllib.error.HTTPError as err:
            payload = err.read()
            self._forward(err.code, dict(err.headers or {}), payload)
        except urllib.error.URLError as err:
            payload = json.dumps(
                {
                    "detail": "Máy chủ tĩnh không kết nối được API tại %s (%s)."
                    % (self.api_base, err.reason)
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    def _forward(self, status: int, headers: dict, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def end_headers(self) -> None:
        # Không cache để sửa code trong lúc lab là thấy ngay.
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def do_GET(self) -> None:
        if self._proxy_request():
            self._proxy()
        else:
            super().do_GET()

    def do_HEAD(self) -> None:
        if self._proxy_request():
            self._proxy()
        else:
            super().do_HEAD()

    def do_POST(self) -> None:
        if self._proxy_request():
            self._proxy()
        else:
            self.send_error(405, "Máy chủ tĩnh chỉ nhận GET/HEAD.")

    def do_OPTIONS(self) -> None:
        if self._proxy_request():
            self.send_response(204)
            self.send_header("Allow", "GET, HEAD, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_error(405, "Máy chủ tĩnh chỉ nhận GET/HEAD.")


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    # Windows PowerShell may expose a legacy code page that cannot print
    # Vietnamese argparse help or runtime messages.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Máy chủ tĩnh cho giao diện LAB3.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Địa chỉ lắng nghe (mặc định 127.0.0.1).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Cổng (mặc định 5500).")
    parser.add_argument("--dir", default=str(HERE), help="Thư mục phục vụ (mặc định thư mục chứa serve.py).")
    parser.add_argument("--api", default=DEFAULT_API, help="Địa chỉ API FastAPI để proxy.")
    parser.add_argument("--no-proxy", action="store_true", help="Không proxy, chỉ phục vụ file tĩnh.")
    parser.add_argument("--no-browser", action="store_true", help="Không tự mở trình duyệt.")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.dir, "index.html")):
        print("Thư mục %s không có index.html." % args.dir)
        return 1

    Handler.api_base = "" if args.no_proxy else args.api.rstrip("/")
    handler = functools.partial(Handler, directory=os.path.abspath(args.dir))
    try:
        httpd = Server((args.host, args.port), handler)
    except OSError as err:
        print("Không mở được cổng %d: %s" % (args.port, err))
        print("Thử cổng khác: python serve.py --port 8080")
        return 1

    url = "http://%s:%d/" % ("localhost" if args.host in ("0.0.0.0", "") else args.host, args.port)
    print("Phục vụ %s" % os.path.abspath(args.dir))
    print("Mở %s (localhost được coi là nguồn tin cậy nên camera chạy được)." % url)
    if Handler.api_base:
        print("Proxy /api và /health -> %s" % Handler.api_base)
        print('Trong giao diện để "Địa chỉ API" trống để dùng cùng origin, tránh lỗi CORS.')
    else:
        print("Không dùng proxy: nhập địa chỉ API đầy đủ trong giao diện và bảo đảm API cho phép CORS.")
    print("Nhấn Ctrl+C để dừng.")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
