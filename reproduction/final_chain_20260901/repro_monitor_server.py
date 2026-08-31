#!/usr/bin/env python3
"""Dependency-free, read-only web server for reproduction quality monitoring."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from repro_quality import build_snapshot, load_json


class ReproductionMonitorHandler(BaseHTTPRequestHandler):
    server_version = "OneReasonReproMonitor/1.0"

    @property
    def app(self) -> "ReproductionMonitorServer":
        return self.server  # type: ignore[return-value]

    def send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, path: Path) -> None:
        static_root = self.app.static_root.resolve()
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if static_root != resolved and static_root not in resolved.parents:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        body = resolved.read_bytes()
        content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self.send_json({"status": "ok", "root": str(self.app.reproduction_root)})
            return
        if parsed.path == "/api/reference":
            self.send_json(load_json(self.app.reference_path))
            return
        if parsed.path == "/api/snapshot":
            try:
                self.send_json(build_snapshot(self.app.reproduction_root, self.app.reference_path))
            except Exception as error:  # Fail visibly without terminating the monitor.
                self.send_json({"status": "error", "error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path in {"", "/"}:
            self.send_static(self.app.static_root / "index.html")
            return
        relative = unquote(parsed.path.removeprefix("/static/")) if parsed.path.startswith("/static/") else unquote(parsed.path.removeprefix("/"))
        self.send_static(self.app.static_root / relative)

    def log_message(self, format: str, *args: object) -> None:
        if not self.app.quiet:
            super().log_message(format, *args)


class ReproductionMonitorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], *, reproduction_root: Path, reference_path: Path, static_root: Path, quiet: bool = False):
        self.reproduction_root = reproduction_root.resolve()
        self.reference_path = reference_path.resolve(strict=True)
        self.static_root = static_root.resolve(strict=True)
        self.quiet = quiet
        super().__init__(address, ReproductionMonitorHandler)


def create_server(host: str, port: int, *, root: Path, reference: Path | None = None, static: Path | None = None, quiet: bool = False) -> ReproductionMonitorServer:
    module_root = Path(__file__).resolve().parent
    return ReproductionMonitorServer(
        (host, port),
        reproduction_root=root,
        reference_path=reference or module_root / "historical_reference.json",
        static_root=static or module_root / "repro_monitor_static",
        quiet=quiet,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--static", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    server = create_server(args.host, args.port, root=args.root, reference=args.reference, static=args.static, quiet=args.quiet)
    print(f"REPRO_MONITOR_READY http://{args.host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
