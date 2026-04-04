from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

BUILD_EXE = Path("dist_portable/Graphik/Graphik.exe")
RUN_PACKAGED_SMOKE = os.environ.get("GRAPHIK_RUN_PACKAGED_SMOKE") == "1"
pytestmark = [
    pytest.mark.skipif(os.name != "nt", reason="Windows-only packaged smoke tests."),
    pytest.mark.skipif(not RUN_PACKAGED_SMOKE, reason="Set GRAPHIK_RUN_PACKAGED_SMOKE=1 to run packaged smoke tests."),
]


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _fetch(url: str, timeout_seconds: float = 1.5) -> tuple[int | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return response.status, response.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, ""


def _wait_for_ready(port: int, timeout_seconds: float = 45.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        health_status, health_body = _fetch(f"http://127.0.0.1:{port}/_stcore/health")
        root_status, root_body = _fetch(f"http://127.0.0.1:{port}/")
        if health_status == 200 and health_body.strip().lower() == "ok" and root_status == 200 and "html" in root_body.lower():
            return True
        time.sleep(0.5)
    return False


@pytest.mark.skipif(not BUILD_EXE.exists(), reason="Packaged Graphik.exe not found.")
def test_packaged_app_starts_and_serves_root() -> None:
    if not _port_available(8501):
        pytest.skip("Port 8501 is already in use.")

    env = os.environ.copy()
    env["GRAPHIK_NO_BROWSER"] = "1"
    proc = subprocess.Popen([str(BUILD_EXE)], cwd=str(BUILD_EXE.parent), env=env)
    try:
        assert _wait_for_ready(8501)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


@pytest.mark.skipif(not BUILD_EXE.exists(), reason="Packaged Graphik.exe not found.")
def test_packaged_app_smoke_export_png_and_pdf(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["GRAPHIK_NO_BROWSER"] = "1"
    result = subprocess.run(
        [str(BUILD_EXE), "--smoke-export", str(tmp_path)],
        cwd=str(BUILD_EXE.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert (tmp_path / "smoke.png").read_bytes().startswith(bytes([137]) + b"PNG")
    assert (tmp_path / "smoke.pdf").read_bytes().startswith(b"%PDF")
