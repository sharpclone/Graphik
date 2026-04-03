"""Desktop launcher for packaged Graphik builds."""

from __future__ import annotations

from pathlib import Path
import os
import socket
import sys
import threading
import time
import webbrowser

from streamlit.web import bootstrap


def runtime_root() -> Path:
    """Return the bundle root or project root."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def find_free_port() -> int:
    """Reserve an ephemeral localhost port for Streamlit."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def open_browser_later(port: int) -> None:
    """Open the local app URL after Streamlit has had a moment to start."""
    if os.environ.get("GRAPHIK_NO_BROWSER") == "1":
        return

    def _worker() -> None:
        time.sleep(1.4)
        webbrowser.open(f"http://127.0.0.1:{port}", new=1)

    threading.Thread(target=_worker, daemon=True).start()


def main() -> None:
    """Start Streamlit app with desktop-friendly defaults."""
    root = runtime_root()
    app_path = root / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Could not find bundled app.py at {app_path}")

    port = int(os.environ.get("GRAPHIK_PORT") or find_free_port())
    open_browser_later(port)

    bootstrap.run(
        str(app_path),
        False,
        [],
        {
            "server.headless": True,
            "server.address": "127.0.0.1",
            "server.port": port,
            "browser.gatherUsageStats": False,
            "global.developmentMode": False,
        },
    )


if __name__ == "__main__":
    main()
