"""Tests for the model pool and the server lifecycle. No GPU needed."""

import sys
import textwrap
import urllib.request

import pytest

from eval_mm.model_pool import ModelSpec, load_pool
from eval_mm.serve import serving, vllm_command


def test_pool_loads_and_ids_are_unique():
    pool = load_pool()
    assert len({spec.id for spec in pool}) == len(pool)
    assert any(spec.status == "active" for spec in pool)


def test_pending_models_say_why():
    assert [s.id for s in load_pool() if s.status == "pending" and not s.note] == []


def test_invalid_entries_are_rejected(tmp_path):
    path = tmp_path / "pool.toml"
    path.write_text('[[models]]\nid = "a/b"\nthinking = "maybe"\n')
    with pytest.raises(ValueError, match="thinking"):
        load_pool(path)
    path.write_text('[[models]]\nid = "a/b"\n\n[[models]]\nid = "a/b"\n')
    with pytest.raises(ValueError, match="Duplicated"):
        load_pool(path)


def test_vllm_command_accepts_a_command_with_arguments():
    spec = ModelSpec(id="org/model", gpus=2, serve_args=["--max-model-len", "8192"])
    assert vllm_command(spec, 8123, "uvx --from vllm==0.33.0 vllm") == [
        "uvx", "--from", "vllm==0.33.0", "vllm",
        "serve", "org/model", "--host", "127.0.0.1", "--port", "8123",
        "--tensor-parallel-size", "2", "--max-model-len", "8192",
    ]  # fmt: skip


FAKE_SERVER = textwrap.dedent(
    """
    import sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

    HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), Handler).serve_forever()
    """
)


def test_serving_starts_and_stops_the_server(tmp_path):
    script = tmp_path / "fake_vllm.py"
    script.write_text(FAKE_SERVER)
    spec = ModelSpec(id="org/model")

    with serving(
        spec,
        port=18766,
        vllm_cmd=f"{sys.executable} {script}",
        log_path=tmp_path / "serve.log",
        startup_timeout=30,
    ) as base_url:
        assert base_url == "http://127.0.0.1:18766/v1"
        assert urllib.request.urlopen("http://127.0.0.1:18766/health", timeout=5).status == 200

    with pytest.raises(OSError):
        urllib.request.urlopen("http://127.0.0.1:18766/health", timeout=2)


def test_serving_reports_a_server_that_exits(tmp_path):
    script = tmp_path / "broken.py"
    script.write_text("import sys; sys.exit(3)")
    with pytest.raises(RuntimeError, match="code 3"):
        with serving(
            ModelSpec(id="org/model"),
            port=18767,
            vllm_cmd=f"{sys.executable} {script}",
            log_path=tmp_path / "serve.log",
            startup_timeout=30,
        ):
            pass
