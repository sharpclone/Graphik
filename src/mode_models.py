"""Typed configuration and analysis result models for application modes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go

from .calculations import ErrorLineResult, LinearFitResult
from .export_utils import PlotTextBlockLayout
from .statistics import DistributionStats
from .ux_models import PlotInfoBoxStatus


@dataclass(frozen=True)
class PlotInfoBoxConfig:
    """User-facing configuration for the plot info box."""

    manual: bool = False
    font_size: int = 12
    x: float | None = None
    y: float | None = None
    max_width: float | None = None
    max_height: float | None = None

    def to_layout(self) -> PlotTextBlockLayout:
        """Convert to the low-level Plotly placement layout."""
        return PlotTextBlockLayout(
            manual=self.manual,
            x=self.x,
            y=self.y,
            max_width=self.max_width,
            max_height=self.max_height,
        )


@dataclass(frozen=True)
class NormalModeConfig:
    """All normal-mode controls collected from the sidebar UI."""

    mapping_mode: str
    use_zero_error: bool
    x_column: str
    y_column: str
    sigma_y_column: str
    use_math_text: bool
    auto_axis_labels: bool
    x_label: str
    y_label: str
    y_axis_type: str
    y_log_decades: int | None
    base_font_size: int
    axis_title_font_size: int
    tick_font_size: int
    annotation_font_size: int
    plot_info_box: PlotInfoBoxConfig
    show_grid: bool
    grid_mode: str
    x_major_divisions: int
    y_major_divisions: int
    minor_per_major: int
    marker_size: float
    error_bar_thickness: float
    error_bar_cap_width: float
    connect_points: bool
    x_tick_decimals: int
    y_tick_decimals: int
    x_range: tuple[float, float] | None
    y_range: tuple[float, float] | None
    fit_model: str
    show_fit_line: bool
    show_error_lines: bool
    extrapolate_lines: bool
    error_line_mode: str
    fit_label: str
    fit_color: str
    show_fit_slope_label: bool
    show_line_equations_on_plot: bool
    show_r2_on_plot: bool
    visible_error_lines_linear: tuple[str, ...]
    visible_error_lines_exp: tuple[str, ...]
    error_color: str
    error_label_max: str
    error_label_min: str
    show_error_slope_label: bool
    show_fit_triangle: bool
    auto_fit_points: bool
    custom_fit_x_a: float | None
    custom_fit_x_b: float | None
    show_error_triangles: bool
    triangle_x_decimals: int
    triangle_y_decimals: int
    auto_line_labels_enabled: bool = True


@dataclass(frozen=True)
class StatisticsModeConfig:
    """All statistics-mode controls collected from the sidebar UI."""

    stats_column: str
    bins: int
    normalize_density: bool
    use_math_text: bool
    auto_axis_labels: bool
    x_label: str
    y_label: str
    base_font_size: int
    axis_title_font_size: int
    tick_font_size: int
    annotation_font_size: int
    plot_info_box: PlotInfoBoxConfig
    show_grid: bool
    x_tick_decimals: int
    y_tick_decimals: int
    show_normal_fit: bool
    show_formula_box: bool
    show_mean_line: bool
    show_std_lines: bool
    show_two_sigma: bool
    show_three_sigma: bool
    histogram_color: str
    fit_color: str
    mean_color: str
    std_color: str


@dataclass(frozen=True)
class PlotContract:
    """Contract for keeping preview and export visually equivalent."""

    preview_figure: go.Figure
    plot_info_lines: tuple[str, ...]
    plot_info_box: PlotInfoBoxConfig
    annotation_font_size: int
    build_export_base_figure: Callable[..., go.Figure]
    plot_info_status: PlotInfoBoxStatus | None = None


@dataclass(frozen=True)
class NormalAnalysisResult:
    """Complete normal-mode analysis state shared by preview, export, and summaries."""

    analysis_df: pd.DataFrame
    x_data: tuple[float, ...]
    y_data: tuple[float, ...]
    sigma_y_data: tuple[float, ...]
    fit_result: LinearFitResult
    fit_prefactor: float | None
    error_line_result: ErrorLineResult | None
    error_line_error: str | None
    error_line_method: str
    fit_triangle_slope: float | None
    plot_contract: PlotContract
    error_triangle_slopes: dict[str, float] = field(default_factory=dict)
    final_slope: str | None = None
    messages: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StatisticsAnalysisResult:
    """Complete statistics-mode analysis state shared by preview, export, and summaries."""

    numeric_values: tuple[float, ...]
    numeric_values_df: pd.DataFrame
    dropped_count: int
    stats_result: DistributionStats
    include_normal_fit: bool
    is_constant_data: bool
    plot_contract: PlotContract
    messages: tuple[tuple[str, str], ...] = field(default_factory=tuple)
