"""Start a model server, evaluate against it, and stop it."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

from .model_pool import ModelSpec


def vllm_command(spec: ModelSpec, port: int, vllm_cmd: str = "vllm") -> list[str]:
    """Command line that serves *spec* with ``vllm serve``.

    *vllm_cmd* may contain arguments, e.g. ``uvx --from vllm==0.33.0 vllm``,
    so vLLM can live in an environment separate from this package.
    """
    return [
        *shlex.split(vllm_cmd),
        "serve",
        spec.id,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(spec.gpus),
        *spec.serve_args,
    ]


def _is_ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def wait_until_ready(
    base_url: str, process: subprocess.Popen, timeout: float, log_path: Path
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server exited with code {process.returncode} before it was ready. See {log_path}"
            )
        if _is_ready(base_url):
            return
        time.sleep(5)
    raise TimeoutError(f"Server was not ready within {timeout:.0f}s. See {log_path}")


def stop(process: subprocess.Popen, grace: float = 30) -> None:
    """Stop the server and the worker processes it started."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@contextmanager
def serving(
    spec: ModelSpec,
    *,
    port: int = 8000,
    vllm_cmd: str = "vllm",
    log_path: Path,
    startup_timeout: float = 3600,
) -> Iterator[str]:
    """Serve *spec* for the duration of the block and yield the API base URL."""
    command = vllm_command(spec, port, vllm_cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting server: {shlex.join(command)} (log: {log_path})")
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            wait_until_ready(f"http://127.0.0.1:{port}", process, startup_timeout, log_path)
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            stop(process)
