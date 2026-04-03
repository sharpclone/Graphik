"""Application-specific exception types."""

from __future__ import annotations

from pathlib import Path


class GraphikError(Exception):
    """Base class for application-specific errors."""


class ExportValidationError(GraphikError):
    """Raised when an export request or figure fails validation."""

    def __init__(self, message: str, *, report_path: Path | None = None) -> None:
        super().__init__(message)
        self.report_path = report_path


class ExportRenderError(GraphikError):
    """Raised when export rendering fails after validation passed."""

    def __init__(self, message: str, *, report_path: Path | None = None) -> None:
        super().__init__(message)
        self.report_path = report_path


class StateRestoreError(GraphikError):
    """Raised when persisted Streamlit session state cannot be restored."""


class StateSaveError(GraphikError):
    """Raised when persisted Streamlit session state cannot be saved."""
