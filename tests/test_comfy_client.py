"""ComfyClient against a small in-process mock of ComfyUI's HTTP API (no GPU)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from maker3d import workflows
from maker3d.comfy_client import (
    ComfyClient,
    ComfyError,
    ComfyExecutionError,
    ComfyValidationError,
    OutputFile,
    find_node,
    output_files,
    with_inputs,
)

PROMPT_ID = "abc123"
DONE = {
    PROMPT_ID: {
        "outputs": {
            "9": {
                "images": [{"filename": "a_00001_.png", "subfolder": "maker3d", "type": "output"}]
            },
            "40": {"3d": [{"filename": "m_00001_.glb", "subfolder": "3d", "type": "output"}]},
        },
        "status": {"status_str": "success", "completed": True, "messages": []},
    }
}


class MockComfy:
    """State shared with the request handler; tests set the responses."""

    def __init__(self):
        self.prompt_status = 200
        self.prompt_response = {"prompt_id": PROMPT_ID, "number": 0, "node_errors": {}}
        self.history_responses = [DONE]  # served in order; the last one repeats
        self.requests = []  # (method, path, query, body)

    def next_history(self):
        if len(self.history_responses) > 1:
            return self.history_responses.pop(0)
        return self.history_responses[0]


@pytest.fixture
def comfy():
    state = MockComfy()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, body, content_type="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            state.requests.append(("GET", url.path, parse_qs(url.query), b""))
            if url.path == f"/history/{PROMPT_ID}":
                self._send(200, state.next_history())
            elif url.path == "/view":
                q = parse_qs(url.query)
                self._send(200, f"bytes of {q['subfolder'][0]}/{q['filename'][0]}".encode(), "x/y")
            elif url.path == "/system_stats":
                self._send(200, {"system": {"comfyui_version": "0.44.0"}, "devices": []})
            else:
                self._send(404, {})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            state.requests.append(("POST", self.path, {}, body))
            if self.path == "/prompt":
                self._send(state.prompt_status, state.prompt_response)
            elif self.path == "/upload/image":
                self._send(200, {"name": "char.png", "subfolder": "", "type": "input"})
            elif self.path in ("/interrupt", "/free"):
                self._send(200, b"", "text/plain")
            else:
                self._send(404, {})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.client = ComfyClient(f"http://127.0.0.1:{server.server_address[1]}/", client_id="test")
    yield state
    server.shutdown()
    server.server_close()


def test_run_success_waits_and_lists_outputs(comfy, tmp_path):
    comfy.history_responses = [{}, {}, DONE]  # queued, running, done
    polls = []
    files = comfy.client.run(
        {"1": {"class_type": "X", "inputs": {}}}, interval=0.01, on_poll=polls.append
    )

    assert len(polls) == 2
    method, path, _, body = comfy.requests[0]
    assert (method, path) == ("POST", "/prompt")
    assert json.loads(body) == {
        "prompt": {"1": {"class_type": "X", "inputs": {}}},
        "client_id": "test",
    }
    assert files == [
        OutputFile("9", "images", "a_00001_.png", "maker3d", "output"),
        OutputFile("40", "3d", "m_00001_.glb", "3d", "output"),
    ]
    dest = comfy.client.download(files[1], tmp_path)
    assert dest.read_bytes() == b"bytes of 3d/m_00001_.glb"


def test_wait_times_out(comfy):
    comfy.history_responses = [{}]
    with pytest.raises(TimeoutError, match=PROMPT_ID):
        comfy.client.wait(PROMPT_ID, timeout=0.2, interval=0.02)


def test_queue_validation_error(comfy):
    comfy.prompt_status = 400
    comfy.prompt_response = {
        "error": {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt outputs failed validation",
            "details": "",
        },
        "node_errors": {
            "1": {
                "errors": [{"message": "Value not in list", "details": "unet_name: 'x' not in []"}],
                "class_type": "UNETLoader",
            }
        },
    }
    with pytest.raises(ComfyValidationError) as exc:
        comfy.client.queue({})
    assert "node 1 (UNETLoader): Value not in list" in str(exc.value)
    assert "unet_name" in str(exc.value)
    assert set(exc.value.node_errors) == {"1"}


def test_partial_node_errors_warn(comfy):
    comfy.prompt_response = {
        "prompt_id": PROMPT_ID,
        "node_errors": {"5": {"errors": [{"message": "bad"}], "class_type": "SaveGLB"}},
    }
    with pytest.warns(UserWarning, match="SaveGLB"):
        assert comfy.client.queue({}) == PROMPT_ID


def test_execution_error(comfy):
    comfy.history_responses = [
        {
            PROMPT_ID: {
                "outputs": {},
                "status": {
                    "status_str": "error",
                    "completed": False,
                    "messages": [
                        ["execution_start", {"prompt_id": PROMPT_ID}],
                        [
                            "execution_error",
                            {
                                "node_id": "12",
                                "node_type": "KSampler",
                                "exception_type": "torch.OutOfMemoryError",
                                "exception_message": "CUDA out of memory.\n",
                            },
                        ],
                    ],
                },
            }
        }
    ]
    with pytest.raises(ComfyExecutionError) as exc:
        comfy.client.wait(PROMPT_ID, interval=0.01)
    assert exc.value.node_id == "12"
    assert (
        str(exc.value) == "node 12 (KSampler) failed: torch.OutOfMemoryError: CUDA out of memory."
    )


def test_interrupted(comfy):
    comfy.history_responses = [
        {
            PROMPT_ID: {
                "status": {
                    "status_str": "error",
                    "messages": [
                        ["execution_interrupted", {"node_id": "3", "node_type": "KSampler"}]
                    ],
                }
            }
        }
    ]
    with pytest.raises(ComfyError, match="interrupted at node 3"):
        comfy.client.wait(PROMPT_ID, interval=0.01)


def test_upload_and_server_control(comfy, tmp_path):
    png = tmp_path / "char.png"
    png.write_bytes(b"\x89PNG fake")
    assert comfy.client.upload_image(png) == "char.png"
    _, path, _, body = comfy.requests[-1]
    assert path == "/upload/image"
    assert b'filename="char.png"' in body and b"\x89PNG fake" in body

    assert comfy.client.system_stats()["system"]["comfyui_version"] == "0.44.0"
    comfy.client.interrupt()
    comfy.client.free()
    assert json.loads(comfy.requests[-1][3]) == {"unload_models": True, "free_memory": True}


def test_with_inputs_by_title():
    wf = workflows.anima_t2i("1girl", seed=1)
    wf2 = with_inputs(wf, "sampler", seed=42, steps=12)
    sampler = find_node(wf2, "sampler")
    assert wf2[sampler]["inputs"]["seed"] == 42
    assert wf2[sampler]["inputs"]["steps"] == 12
    assert wf[sampler]["inputs"]["seed"] == 1  # original untouched
    with pytest.raises(KeyError):
        find_node(wf, "nope")
    wf2["99"] = {"class_type": "X", "inputs": {}, "_meta": {"title": "sampler"}}
    with pytest.raises(ValueError, match="several"):
        find_node(wf2, "sampler")


def test_output_files_ignores_non_file_entries():
    entry = {"outputs": {"7": {"text": ["hello"], "images": [{"filename": "x.png"}], "flag": True}}}
    assert output_files(entry) == [OutputFile("7", "images", "x.png", "", "output")]
