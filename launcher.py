"""Desktop launcher for packaged Graphik builds."""

from __future__ import annotations

import argparse
import ctypes
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

APP_NAME = "Graphik"
DEFAULT_PORT = 8501
SERVER_WAIT_SECONDS = 30.0
HEALTH_PATH = "_stcore/health"


def runtime_root() -> Path:
    """Return the bundle root or project root."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def app_url(port: int) -> str:
    """Return the local application URL."""
    return f"http://127.0.0.1:{port}"


def health_url(port: int) -> str:
    """Return the health-check URL."""
    return f"{app_url(port)}/{HEALTH_PATH}"


def port_is_available(port: int) -> bool:
    """Return True when localhost port can be bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def wait_for_server(port: int, timeout_seconds: float = SERVER_WAIT_SECONDS) -> bool:
    """Poll localhost until the Streamlit server accepts TCP connections."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def fetch_url(url: str, timeout_seconds: float = 1.5) -> tuple[int | None, str]:
    """Fetch a local URL and return status code plus response body."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="ignore")
            return response.status, body
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, ""


def healthcheck_ok(port: int, timeout_seconds: float = 1.5) -> bool:
    """Return True if the Streamlit health endpoint responds with ok."""
    status, body = fetch_url(health_url(port), timeout_seconds=timeout_seconds)
    return status == 200 and body.strip().lower() == "ok"


def root_ready(port: int, timeout_seconds: float = 1.5) -> bool:
    """Return True if the app root serves the Streamlit shell page."""
    status, body = fetch_url(app_url(port), timeout_seconds=timeout_seconds)
    if status != 200:
        return False
    body_lower = body.lower()
    return "streamlit" in body_lower or "graphik" in body_lower or "<html" in body_lower


def graphik_is_ready(port: int) -> bool:
    """Return True if a Graphik instance seems ready on the port."""
    return healthcheck_ok(port) and root_ready(port)


def open_browser(port: int) -> None:
    """Open the local app URL immediately unless disabled."""
    if os.environ.get("GRAPHIK_NO_BROWSER") == "1":
        return
    webbrowser.open(app_url(port), new=1)


def open_browser_later(port: int) -> None:
    """Open the local app URL once Graphik is actually ready."""
    if os.environ.get("GRAPHIK_NO_BROWSER") == "1":
        return

    def _worker() -> None:
        if wait_for_server(port):
            deadline = time.time() + SERVER_WAIT_SECONDS
            while time.time() < deadline:
                if graphik_is_ready(port):
                    webbrowser.open(app_url(port), new=1)
                    return
                time.sleep(0.25)

    threading.Thread(target=_worker, daemon=True).start()


def show_error_dialog(message: str) -> None:
    """Show a friendly Windows error message for packaged builds."""
    if os.name == "nt":
        ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10)
    else:
        print(message, file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher-only command-line flags."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke-export", dest="smoke_export_dir")
    parser.add_argument("--port", dest="port", type=int)
    return parser.parse_known_args(argv)


def choose_requested_port(cli_port: int | None = None) -> int:
    """Return the configured port, defaulting to 8501."""
    if cli_port is not None:
        return int(cli_port)
    requested = os.environ.get("GRAPHIK_PORT")
    if requested:
        return int(requested)
    return DEFAULT_PORT


def run_smoke_export(output_dir: Path) -> None:
    """Render simple PNG/PDF smoke exports without launching the server."""
    import plotly.graph_objects as go

    from src.plotting import figure_to_image_bytes

    output_dir.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[50, 100, 150, 200],
            y=[0.846, 1.4641, 2.1609, 2.7556],
            mode="lines+markers",
            error_y={"type": "data", "array": [0.0129, 0.0242, 0.0176, 0.0332], "visible": True},
            name="Messwerte",
        )
    )
    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=800,
        xaxis_title="m [g]",
        yaxis_title="T^2 [s^2]",
        font={"size": 14},
        margin={"l": 70, "r": 30, "t": 60, "b": 60},
    )
    for fmt in ("png", "pdf"):
        data = figure_to_image_bytes(fig, fmt, width=1100, height=800, scale=1.0, base_dpi=150)
        (output_dir / f"smoke.{fmt}").write_bytes(data)


def main(argv: list[str] | None = None) -> None:
    """Start Streamlit app with desktop-friendly defaults."""
    args, _unknown = parse_args(argv)
    root = runtime_root()
    app_path = root / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Could not find bundled app.py at {app_path}")

    os.chdir(root)
    os.environ.setdefault("GRAPHIK_RUNTIME", "packaged")
    if args.smoke_export_dir:
        run_smoke_export(Path(args.smoke_export_dir))
        return

    port = choose_requested_port(args.port)

    if graphik_is_ready(port):
        open_browser(port)
        return

    if not port_is_available(port):
        if wait_for_server(port, timeout_seconds=10.0) and graphik_is_ready(port):
            open_browser(port)
            return
        show_error_dialog(
            f"Port {port} is already in use by another application.\n\n"
            f"Close that application or set GRAPHIK_PORT to a different port."
        )
        return

    open_browser_later(port)

    from streamlit.web import bootstrap

    flag_options = {
        "server.headless": True,
        "server.address": "127.0.0.1",
        "server.port": port,
        "browser.gatherUsageStats": False,
        "global.developmentMode": False,
        "theme.base": "light",
        "theme.backgroundColor": "#FFFFFF",
        "theme.secondaryBackgroundColor": "#F5F7FB",
        "theme.textColor": "#1F2937",
        "theme.primaryColor": "#0EA5C6",
    }

    bootstrap.load_config_options(flag_options)
    bootstrap.run(
        str(app_path),
        False,
        [],
        flag_options,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
