"""Structured logging helpers for Graphik."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

LOG_DIR = Path('.streamlit') / 'logs'
LOG_PATH = LOG_DIR / 'graphik.log'


def runtime_mode() -> str:
    """Return whether Graphik is running in dev or packaged mode."""
    if os.environ.get('GRAPHIK_RUNTIME') == 'packaged':
        return 'packaged'
    return 'packaged' if getattr(sys, 'frozen', False) else 'dev'


def configure_logging(level: int = logging.INFO) -> None:
    """Configure Graphik file logging once per process."""
    logger = logging.getLogger('graphik')
    if logger.handlers:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


def log_event(logger: logging.Logger, level: int, event: str, **context: Any) -> None:
    """Write one structured key-value log event."""
    payload = {"event": event, "runtime_mode": runtime_mode(), **context}
    logger.log(level, json.dumps(payload, sort_keys=True, default=str))
