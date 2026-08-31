from __future__ import annotations

import json
import sys
import threading
import urllib.request
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from repro_monitor_server import create_server  # noqa: E402


def test_monitor_serves_health_snapshot_and_frontend(tmp_path: Path) -> None:
    server = create_server("127.0.0.1", 0, root=tmp_path, quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        health = json.load(urllib.request.urlopen(base + "/healthz", timeout=3))
        snapshot = json.load(urllib.request.urlopen(base + "/api/snapshot", timeout=3))
        html = urllib.request.urlopen(base + "/", timeout=3).read().decode("utf-8")
        css = urllib.request.urlopen(base + "/static/styles.css", timeout=3).read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert health["status"] == "ok"
    assert snapshot["summary"]["stage_count"] == 4
    assert "四阶段复现质量监控" in html
    assert ".stage-rail" in css


def test_static_path_traversal_is_rejected(tmp_path: Path) -> None:
    server = create_server("127.0.0.1", 0, root=tmp_path, quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/static/../../historical_reference.json"
    try:
        try:
            urllib.request.urlopen(url, timeout=3)
            status = 200
        except urllib.error.HTTPError as error:
            status = error.code
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert status in {403, 404}
