import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from maker3d.runtime import system_stats, wait_ready

STATS = {"system": {"comfyui_version": "0.44.0"}, "devices": []}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/system_stats":
            body = json.dumps(STATS).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_comfy():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_system_stats_reads_json(fake_comfy):
    assert system_stats(fake_comfy) == STATS


def test_system_stats_none_when_down():
    assert system_stats("http://127.0.0.1:9", timeout=0.5) is None


def test_wait_ready_times_out_fast():
    with pytest.raises(TimeoutError):
        wait_ready("http://127.0.0.1:9", timeout=0.2, interval=0.05)


def test_wait_ready_returns_stats(fake_comfy):
    assert wait_ready(fake_comfy, timeout=5)["system"]["comfyui_version"] == "0.44.0"
