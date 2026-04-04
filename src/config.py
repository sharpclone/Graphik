"""Application configuration constants for the Graphik plotting tool."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

APP_TITLE = "Graphik"


def runtime_root() -> Path:
    """Return the project root for source runs and bundled executables."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = runtime_root()
PRESENT_DIR = PROJECT_ROOT / "Present"
LOGO_WHITE_PATH = PRESENT_DIR / "logo-white.png"
LOGO_DARKBLUE_PATH = PRESENT_DIR / "logo-darkblue.png"
FAVICON_PATH = PRESENT_DIR / "favicon.png"
DEFAULT_X_LABEL = "x"
DEFAULT_Y_LABEL = "y"

DEFAULT_POINT_COLOR = "black"
DEFAULT_FIT_COLOR = "#2ca02c"  # green
DEFAULT_ERROR_LINE_COLOR = "#d62728"  # red


SAMPLE_DATA = pd.DataFrame(
    {
        "m": [50, 100, 150, 200, 250],
        "y": [0.8464, 1.4641, 2.1609, 2.7556, 3.4969],
        "sigma_y": [0.01288, 0.0242, 0.01764, 0.0332, 0.0374],
    }
)

SUMMARY_FLOAT_PRECISION = 6
PLOT_FLOAT_PRECISION = 4
