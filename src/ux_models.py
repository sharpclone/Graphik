"""Shared UX-facing models for import, validation, and plot status."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class ProblemItem:
    """One user-facing validation or availability message."""

    severity: str
    title: str
    detail: str
    code: str = ""
    blocking: bool = False
    related_fields: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PlotInfoBoxStatus:
    """Resolved placement status for the plot info box."""

    manual: bool
    requested_font_size: int
    final_font_size: int
    wrapped: bool
    downscaled: bool
    x: float
    y: float
    width: float
    height: float
    wrap_width: int


@dataclass(frozen=True)
class ImportWizardResult:
    """Resolved import-wizard state for one uploaded table."""

    raw_df: pd.DataFrame
    preview_df: pd.DataFrame
    filename: str
    signature: str
    header_mode: str
    header_row_index: int | None
    suggested_x_column: str | None
    suggested_y_column: str | None
    suggested_sigma_y_column: str | None
    suggested_stats_column: str | None
    problems: tuple[ProblemItem, ...] = field(default_factory=tuple)

