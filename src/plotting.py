"""Plotly plotting utilities for physics lab style diagrams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .geometry import Point, right_triangle_corner, triangle_deltas
from .ui_helpers import to_plot_math_text


@dataclass(frozen=True)
class LineStyle:
    """Visual style definition for lines."""

    color: str
    dash: str
    width: float
    label: str


@dataclass(frozen=True)
class PlotStyle:
    """General visual style configuration."""

    x_label: str
    y_label: str
    show_grid: bool
    measured_points_label: str = "Measured points"
    y_axis_type: str = "linear"  # "linear" or "log"
    y_log_decades: int | None = None
    connect_points: bool = False
    marker_size: float = 7.0
    error_bar_thickness: float = 1.8
    error_bar_cap_width: float = 6.0
    x_tick_decimals: int = 2
    y_tick_decimals: int = 2
    base_font_size: int = 14
    axis_title_font_size: int = 16
    tick_font_size: int = 12
    x_tick0: float | None = None
    y_tick0: float | None = None
    x_major_dtick: float | None = None
    y_major_dtick: float | str | None = None
    show_minor_grid: bool = False
    x_minor_dtick: float | None = None
    y_minor_dtick: float | str | None = None
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None


def visible_x_range(x: np.ndarray, custom: tuple[float, float] | None = None) -> tuple[float, float]:
    """Determine x-range used for drawing lines and auto points."""
    if custom is not None:
        x0, x1 = custom
        if x1 <= x0:
            raise ValueError("x-axis max must be greater than min.")
        return (x0, x1)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max == x_min:
        pad = max(1.0, abs(x_min) * 0.05)
        return (x_min - pad, x_max + pad)
    pad = (x_max - x_min) * 0.05
    return (x_min - pad, x_max + pad)


def _effective_log_y_bounds(
    df: pd.DataFrame,
    style: PlotStyle,
) -> tuple[float, float] | None:
    """Return effective positive y-bounds used for a log axis."""
    y_positive = df["y"][df["y"] > 0]
    if y_positive.empty:
        return None

    if style.y_log_decades is not None and style.y_log_decades > 0:
        y_max_raw = float(np.max(y_positive))
        y_max = float(10.0 ** np.ceil(np.log10(y_max_raw)))
        y_min = y_max / (10.0 ** int(style.y_log_decades))
        return (y_min, y_max)

    if style.y_range is not None:
        y0, y1 = float(style.y_range[0]), float(style.y_range[1])
        y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)
        if y_max <= 0:
            return None
        y_min = max(y_min, np.finfo(float).eps)
        return (y_min, y_max)

    return (float(np.min(y_positive)), float(np.max(y_positive)))


def _add_semilog_paper_gridlines(
    fig: go.Figure,
    y_min: float,
    y_max: float,
) -> None:
    """
    Add explicit horizontal log-paper gridlines (major + minor) for readability.

    This is a renderer-agnostic fallback when native Plotly log-minor grids are sparse.
    """
    if y_min <= 0 or y_max <= 0 or y_max <= y_min:
        return

    exp_min = int(np.floor(np.log10(y_min)))
    exp_max = int(np.ceil(np.log10(y_max)))
    eps = y_max * 1e-12
    shapes: list[dict[str, object]] = []

    for exponent in range(exp_min, exp_max + 1):
        decade = 10.0**exponent
        for multiplier in range(1, 10):
            y_val = float(multiplier) * decade
            if y_val < y_min - eps or y_val > y_max + eps:
                continue
            is_major = multiplier == 1
            shapes.append(
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0.0,
                    "x1": 1.0,
                    "yref": "y",
                    "y0": y_val,
                    "y1": y_val,
                    "layer": "below",
                    "line": {
                        "color": "rgba(120,120,120,0.32)" if is_major else "rgba(120,120,120,0.16)",
                        "width": 1.0 if is_major else 0.8,
                    },
                }
            )

    if shapes:
        existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        fig.update_layout(shapes=(existing_shapes + shapes))


def _log_tick_values(y_min: float, y_max: float, digits: list[int]) -> list[float]:
    """Build readable log tick values within [y_min, y_max]."""
    if y_min <= 0 or y_max <= 0 or y_max <= y_min:
        return []
    exp_min = int(np.floor(np.log10(y_min)))
    exp_max = int(np.ceil(np.log10(y_max)))
    ticks: list[float] = []
    for exponent in range(exp_min, exp_max + 1):
        decade = 10.0**exponent
        for digit in digits:
            value = float(digit) * decade
            if y_min <= value <= y_max:
                ticks.append(value)
    return sorted(set(ticks))


def create_base_figure(df: pd.DataFrame, style: PlotStyle) -> go.Figure:
    """Create base scatter plot with vertical error bars."""
    fig = go.Figure()
    mode = "markers+lines" if style.connect_points else "markers"
    sigma_y = df["sigma_y"].to_numpy(dtype=float)
    error_y: dict[str, object] = {
        "type": "data",
        "array": sigma_y,
        "visible": True,
        "color": "black",
        "thickness": float(style.error_bar_thickness),
        "width": float(style.error_bar_cap_width),
    }
    if style.y_axis_type == "log":
        # On log axis, keep lower bars strictly positive to avoid render artifacts.
        y_vals = df["y"].to_numpy(dtype=float)
        min_floor = np.finfo(float).eps
        lower = np.minimum(sigma_y, np.maximum(y_vals - min_floor, 0.0))
        error_y["symmetric"] = False
        error_y["arrayminus"] = lower

    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode=mode,
            marker={"color": "black", "size": float(style.marker_size)},
            line={"color": "black", "width": 1.2},
            error_y=error_y,
            name=style.measured_points_label,
        )
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title=style.x_label,
        yaxis_title=style.y_label,
        font={"size": int(style.base_font_size)},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
    )
    fig.update_xaxes(
        showgrid=style.show_grid,
        tickformat=f".{int(style.x_tick_decimals)}f",
        title_font={"size": int(style.axis_title_font_size)},
        tickfont={"size": int(style.tick_font_size)},
        tick0=style.x_tick0,
        dtick=style.x_major_dtick,
        gridcolor="rgba(120,120,120,0.32)",
    )
    y_axis_kwargs: dict[str, object] = {
        "showgrid": style.show_grid,
        "title_font": {"size": int(style.axis_title_font_size)},
        "tickfont": {"size": int(style.tick_font_size)},
        "tick0": style.y_tick0,
        "dtick": style.y_major_dtick,
        "gridcolor": "rgba(120,120,120,0.32)",
    }
    if style.y_axis_type == "log":
        y_axis_kwargs["type"] = "log"
        if style.y_major_dtick is None:
            y_axis_kwargs["dtick"] = "D2"
        y_axis_kwargs["tickformat"] = "~g"
        y_axis_kwargs["minorloglabels"] = "complete"
        y_axis_kwargs["exponentformat"] = "power"
    else:
        y_axis_kwargs["tickformat"] = f".{int(style.y_tick_decimals)}f"

    fig.update_yaxes(**y_axis_kwargs)

    if style.show_grid and style.show_minor_grid:
        if style.x_minor_dtick is not None:
            fig.update_xaxes(
                minor={
                    "showgrid": True,
                    "dtick": style.x_minor_dtick,
                    "tick0": style.x_tick0,
                    "gridcolor": "rgba(120,120,120,0.16)",
                }
            )
        if style.y_minor_dtick is not None:
            fig.update_yaxes(
                minor={
                    "showgrid": True,
                    "dtick": style.y_minor_dtick,
                    "tick0": style.y_tick0,
                    "gridcolor": "rgba(120,120,120,0.16)",
                }
            )

    if style.x_range is not None:
        fig.update_xaxes(range=list(style.x_range))
    if style.y_axis_type == "log" and style.y_log_decades is not None and style.y_log_decades > 0:
        y_positive = df["y"][df["y"] > 0]
        if not y_positive.empty:
            y_max_raw = float(np.max(y_positive))
            y_max = float(10.0 ** np.ceil(np.log10(y_max_raw)))
            y_min = y_max / (10.0 ** int(style.y_log_decades))
            fig.update_yaxes(range=[np.log10(y_min), np.log10(y_max)], dtick=1)
    elif style.y_range is not None:
        if style.y_axis_type == "log":
            y0, y1 = style.y_range
            fig.update_yaxes(range=[np.log10(float(y0)), np.log10(float(y1))])
        else:
            fig.update_yaxes(range=list(style.y_range))

    if style.y_axis_type == "log":
        log_bounds = _effective_log_y_bounds(df, style)
        if log_bounds is not None:
            y_min, y_max = log_bounds
            if style.y_major_dtick == "D1":
                digits = list(range(1, 10))
            elif style.y_major_dtick == 1 or style.y_major_dtick == 1.0:
                digits = [1]
            else:
                digits = [1, 2, 5]
            tick_values = _log_tick_values(y_min=y_min, y_max=y_max, digits=digits)
            if tick_values:
                fig.update_yaxes(
                    tickmode="array",
                    tickvals=tick_values,
                    ticktext=[f"{value:g}" for value in tick_values],
                )

    if (
        style.y_axis_type == "log"
        and style.show_grid
        and style.show_minor_grid
        and style.y_minor_dtick == "D1"
    ):
        log_bounds = _effective_log_y_bounds(df, style)
        if log_bounds is not None:
            _add_semilog_paper_gridlines(fig, y_min=log_bounds[0], y_max=log_bounds[1])

    return fig


def add_line(
    fig: go.Figure,
    slope: float,
    intercept: float,
    x_bounds: tuple[float, float],
    line_style: LineStyle,
) -> None:
    """Add a line segment over given x bounds."""
    x0, x1 = x_bounds
    xs = np.array([x0, x1], dtype=float)
    ys = slope * xs + intercept

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": line_style.color, "width": line_style.width, "dash": line_style.dash},
            name=line_style.label,
        )
    )


def add_exponential_line(
    fig: go.Figure,
    slope_log: float,
    intercept_log: float,
    x_bounds: tuple[float, float],
    line_style: LineStyle,
    num_points: int = 300,
) -> None:
    """Add exponential curve y = exp(slope_log*x + intercept_log)."""
    x0, x1 = x_bounds
    xs = np.linspace(float(x0), float(x1), int(max(50, num_points)))
    ys = np.exp(slope_log * xs + intercept_log)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": line_style.color, "width": line_style.width, "dash": line_style.dash},
            name=line_style.label,
        )
    )


def add_slope_triangle(
    fig: go.Figure,
    a: Point,
    b: Point,
    color: str,
    label_prefix: str,
    horizontal_symbol: str = "Δx",
    vertical_symbol: str = "Δy",
    use_latex: bool = False,
    x_decimals: int = 2,
    y_decimals: int = 2,
    font_size: int = 12,
    annotate: bool = True,
) -> float:
    """Add right-triangle legs used to visualize slope and return slope from A/B."""
    corner = right_triangle_corner(a, b)
    delta_x, delta_y, slope = triangle_deltas(a, b)

    fig.add_trace(
        go.Scatter(
            x=[a.x, corner.x],
            y=[a.y, corner.y],
            mode="lines",
            line={"color": color, "width": 2, "dash": "dot"},
            name=f"{label_prefix} {horizontal_symbol}",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[corner.x, b.x],
            y=[corner.y, b.y],
            mode="lines",
            line={"color": color, "width": 2, "dash": "dot"},
            name=f"{label_prefix} {vertical_symbol}",
            showlegend=False,
        )
    )

    if annotate:
        delta_x_disp = abs(delta_x)
        delta_y_disp = abs(delta_y)
        dx_text = to_plot_math_text(
            f"{horizontal_symbol} = {delta_x_disp:.{int(x_decimals)}f}",
            use_latex,
        )
        dy_text = to_plot_math_text(
            f"{vertical_symbol} = {delta_y_disp:.{int(y_decimals)}f}",
            use_latex,
        )
        fig.add_annotation(
            x=(a.x + corner.x) / 2.0,
            y=a.y,
            text=dx_text,
            showarrow=False,
            yshift=-12,
            font={"color": color, "size": int(font_size)},
        )
        fig.add_annotation(
            x=b.x,
            y=(corner.y + b.y) / 2.0,
            text=dy_text,
            showarrow=False,
            xshift=28,
            font={"color": color, "size": int(font_size)},
        )

    return slope


def add_line_slope_annotation(
    fig: go.Figure,
    slope: float,
    intercept: float,
    x_position: float,
    color: str,
    label_prefix: str,
    decimals: int = 6,
    font_size: int = 12,
    xshift: int = -8,
    yshift: int = 10,
) -> None:
    """Annotate line slope at a chosen x position."""
    y_position = slope * x_position + intercept
    fig.add_annotation(
        x=x_position,
        y=y_position,
        text=f"{label_prefix}: k = {slope:.{int(decimals)}g}",
        showarrow=False,
        xshift=int(xshift),
        yshift=int(yshift),
        xanchor="right",
        font={"color": color, "size": int(font_size)},
        bgcolor="rgba(255,255,255,0.65)",
    )


def figure_to_image_bytes(
    fig: go.Figure,
    fmt: str,
    width: int | None = None,
    height: int | None = None,
    scale: float = 1.0,
) -> bytes:
    """Export figure to image bytes for download."""
    if fmt not in {"png", "svg", "pdf"}:
        raise ValueError("Supported formats are 'png', 'svg', and 'pdf'.")
    kwargs: dict[str, int | float | str] = {"format": fmt, "scale": float(scale)}
    if width is not None:
        kwargs["width"] = int(width)
    if height is not None:
        kwargs["height"] = int(height)
    return fig.to_image(**kwargs)
