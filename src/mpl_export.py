"""Matplotlib-based export backend for Plotly preview figures.

Supported export contract:
- scatter traces with markers/lines and optional `error_x` / `error_y`
- vertical bar traces
- shape types `line`, `rect`, `circle`
- text annotations without arrows
- linear/log x and y axes
- Plotly legend/title/axis-title metadata used by Graphik

Unsupported preview-only elements fail fast during export validation so we do
not silently generate incomplete PNG/SVG/PDF artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import html
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, MultipleLocator
from matplotlib.transforms import Bbox, blended_transform_factory, offset_copy
import numpy as np
import plotly.graph_objects as go

from .errors import ExportValidationError


_PLOTLY_DASH_MAP: dict[str, Any] = {
    "solid": "-",
    "dash": "--",
    "dot": ":",
    "dashdot": "-.",
    "longdash": (0, (8, 4)),
    "longdashdot": (0, (8, 4, 2, 4)),
}

_CSS_PX_TO_PT = 72.0 / 96.0
SUPPORTED_TRACE_TYPES = frozenset({"scatter", "bar"})
SUPPORTED_SHAPE_TYPES = frozenset({"line", "rect", "circle"})
SUPPORTED_AXIS_TYPES = frozenset({"", "linear", "log", "-", "none"})


@dataclass(frozen=True)
class ExportRenderSummary:
    """Invariant summary used by export parity tests."""

    expected_trace_count: int
    rendered_trace_count: int
    expected_shape_count: int
    rendered_shape_count: int
    expected_annotation_count: int
    rendered_annotation_count: int
    legend_present: bool
    xaxis_title: str
    yaxis_title: str
    plot_title: str
    xscale: str
    yscale: str


def supported_export_contract_lines() -> tuple[str, ...]:
    """Return a concise product-facing description of the export contract."""
    return (
        "scatter (markers/lines, error_x/error_y)",
        "bar",
        "shapes: line/rect/circle",
        "annotations without arrows",
        "axes: linear/log",
    )


def _font_px_to_pt(value: Any, default: float = 12.0) -> float:
    """Convert Plotly/CSS pixel font sizes to Matplotlib point sizes."""
    try:
        px = float(default if value is None else value)
    except (TypeError, ValueError):
        px = float(default)
    return max(1.0, px * _CSS_PX_TO_PT)


def _linewidth_px_to_pt(value: Any, default: float = 1.0) -> float:
    """Convert Plotly/CSS pixel widths to Matplotlib point widths."""
    try:
        px = float(default if value is None else value)
    except (TypeError, ValueError):
        px = float(default)
    return max(0.6, px * _CSS_PX_TO_PT)


def _to_float_array(values: Any) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return np.array([], dtype=float)
    if arr.ndim == 0:
        return np.array([float(arr)], dtype=float)
    return arr.astype(float)


def _to_string_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, np.ndarray)):
        return [str(value) for value in values]
    return [str(values)]


def _html_to_mpl_text(text: Any) -> str:
    out = html.unescape(str(text) if text is not None else "")
    out = re.sub(r"(?i)<br\s*/?>", "\n", out)
    out = re.sub(r"<sup>(.*?)</sup>", lambda match: f"$^{{{match.group(1)}}}$", out)
    out = re.sub(r"<sub>(.*?)</sub>", lambda match: f"$_{{{match.group(1)}}}$", out)
    out = re.sub(r"</?[^>]+>", "", out)
    return out

def _is_visible_trace(trace: Any) -> bool:
    return getattr(trace, "visible", True) not in (False, "legendonly")


def _visible_annotations(fig: go.Figure) -> list[Any]:
    annotations: list[Any] = []
    for annotation in fig.layout.annotations or []:
        text = _html_to_mpl_text(getattr(annotation, "text", ""))
        if text.strip():
            annotations.append(annotation)
    return annotations


def _visible_supported_shapes(fig: go.Figure) -> list[Any]:
    return [
        shape
        for shape in (fig.layout.shapes or [])
        if str(getattr(shape, "type", "")).lower() in SUPPORTED_SHAPE_TYPES
    ]


def validate_supported_export_features(fig: go.Figure) -> None:
    """Fail fast when the preview contains elements the Matplotlib backend cannot reproduce."""
    unsupported: list[str] = []

    for index, trace in enumerate(fig.data, start=1):
        if not _is_visible_trace(trace):
            continue
        trace_type = str(getattr(trace, "type", "scatter")).lower()
        if trace_type not in SUPPORTED_TRACE_TYPES:
            unsupported.append(f"trace {index}: unsupported type '{trace_type}'")
            continue
        if trace_type == "scatter":
            mode = str(getattr(trace, "mode", "markers") or "markers").lower()
            if "text" in mode and not any(token in mode for token in ("markers", "lines")):
                unsupported.append(f"trace {index}: scatter text-only mode is not supported")
            fill = str(getattr(trace, "fill", "none") or "none").lower()
            if fill not in {"none", ""}:
                unsupported.append(f"trace {index}: scatter fill '{fill}' is not supported")
        if trace_type == "bar":
            orientation = str(getattr(trace, "orientation", "v") or "v").lower()
            if orientation == "h":
                unsupported.append(f"trace {index}: horizontal bar traces are not supported")

    for axis_name in ("xaxis", "yaxis"):
        axis = getattr(fig.layout, axis_name, None)
        axis_type = str(getattr(axis, "type", "linear") or "").lower()
        if axis_type not in SUPPORTED_AXIS_TYPES:
            unsupported.append(f"{axis_name}: axis type '{axis_type}' is not supported")

    for index, shape in enumerate(fig.layout.shapes or [], start=1):
        shape_type = str(getattr(shape, "type", "")).lower()
        if shape_type not in SUPPORTED_SHAPE_TYPES:
            unsupported.append(f"shape {index}: unsupported type '{shape_type}'")

    for index, annotation in enumerate(fig.layout.annotations or [], start=1):
        if bool(getattr(annotation, "showarrow", False)):
            unsupported.append(f"annotation {index}: arrows are not supported in export")

    if unsupported:
        contract = ", ".join(supported_export_contract_lines())
        details = "; ".join(unsupported[:6])
        if len(unsupported) > 6:
            details += f"; ... (+{len(unsupported) - 6} more)"
        raise ExportValidationError(
            f"Unsupported feature in export backend: {details}. Supported export contract: {contract}."
        )


def _parse_color(value: Any, default: Any = "black") -> Any:
    if value is None:
        return default
    if isinstance(value, (tuple, list)):
        return value
    text = str(value).strip()
    if not text:
        return default

    rgba_match = re.fullmatch(
        r"rgba?\(([^,]+),([^,]+),([^,]+)(?:,([^,]+))?\)",
        text,
    )
    if rgba_match:
        red = float(rgba_match.group(1)) / 255.0
        green = float(rgba_match.group(2)) / 255.0
        blue = float(rgba_match.group(3)) / 255.0
        alpha = float(rgba_match.group(4)) if rgba_match.group(4) is not None else 1.0
        return (red, green, blue, alpha)

    try:
        return mcolors.to_rgba(text)
    except ValueError:
        return default


def _dash_to_mpl(dash: Any) -> Any:
    if dash is None:
        return "-"
    return _PLOTLY_DASH_MAP.get(str(dash).lower(), "-")


def _soften_grid_style(color: Any, linewidth: float, *, minor: bool = False) -> tuple[Any, float]:
    """Make exported gridlines visually closer to the lighter Plotly preview."""
    rgba = _parse_color(color, default=(0.75, 0.75, 0.75, 0.3))
    if isinstance(rgba, tuple) and len(rgba) == 4:
        red, green, blue, alpha = rgba
    elif isinstance(rgba, tuple) and len(rgba) == 3:
        red, green, blue = rgba
        alpha = 1.0
    else:
        red, green, blue, alpha = (0.75, 0.75, 0.75, 0.3)

    # Vector export renders low-alpha grid strokes more assertively than the browser preview.
    # Blend them toward white and reduce alpha/width slightly to keep the same visual weight.
    blend = 0.48 if minor else 0.38
    alpha_scale = 0.58 if minor else 0.68
    width_scale = 0.72 if minor else 0.78
    softened = (
        red + ((1.0 - red) * blend),
        green + ((1.0 - green) * blend),
        blue + ((1.0 - blue) * blend),
        max(0.04, min(1.0, alpha * alpha_scale)),
    )
    return softened, max(0.25, float(linewidth) * width_scale)


def _is_gridline_shape(shape: Any, color: Any) -> bool:
    """Detect explicit fallback gridline shapes added for semilog paper styling."""
    shape_type = str(getattr(shape, "type", "")).lower()
    if shape_type != "line":
        return False
    if str(getattr(shape, "layer", "")).lower() != "below":
        return False
    xref = str(getattr(shape, "xref", ""))
    yref = str(getattr(shape, "yref", ""))
    if not ((xref == "paper" and yref in {"y", "y2"}) or (yref == "paper" and xref in {"x", "x2"})):
        return False
    rgba = _parse_color(color, default=None)
    if not isinstance(rgba, tuple) or len(rgba) < 3:
        return False
    red, green, blue = rgba[:3]
    alpha = rgba[3] if len(rgba) == 4 else 1.0
    is_gray = max(abs(red - green), abs(green - blue), abs(red - blue)) <= 0.02
    return is_gray and alpha <= 0.45


def _marker_size_to_points(size: Any, default: float = 7.0) -> float:
    try:
        px = float(size if size is not None else default)
    except (TypeError, ValueError):
        px = float(default)
    return max(1.5, px * 0.75)


def _error_arrays(trace: go.Scatter, axis_name: str, length: int) -> tuple[np.ndarray, np.ndarray] | None:
    err = getattr(trace, f"error_{axis_name}", None)
    if err is None or not bool(getattr(err, "visible", False)):
        return None
    plus = _to_float_array(getattr(err, "array", None))
    if plus.size == 0:
        return None
    if plus.size == 1 and length > 1:
        plus = np.full(length, float(plus[0]), dtype=float)
    plus = plus[:length]

    minus = _to_float_array(getattr(err, "arrayminus", None))
    if minus.size == 0:
        minus = plus.copy()
    elif minus.size == 1 and length > 1:
        minus = np.full(length, float(minus[0]), dtype=float)
    else:
        minus = minus[:length]
    return minus, plus


def _axis_range(axis: Any, is_log: bool) -> tuple[float, float] | None:
    axis_range = getattr(axis, "range", None)
    if axis_range is None or len(axis_range) != 2:
        return None
    if axis_range[0] is None or axis_range[1] is None:
        return None
    start = float(axis_range[0])
    end = float(axis_range[1])
    if is_log:
        start = float(10.0**start)
        end = float(10.0**end)
    return (start, end)


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _make_multiple_locator(dtick: Any, tick0: Any) -> MultipleLocator | None:
    base = _safe_float(dtick)
    if base is None or base <= 0:
        return None
    offset = _safe_float(tick0)
    return MultipleLocator(base=base, offset=0.0 if offset is None else offset)


def _apply_axis_format(ax: Any, axis: Any, *, which: str) -> None:
    is_y = which == "y"
    is_log = str(getattr(axis, "type", "linear")).lower() == "log"
    mpl_axis = ax.yaxis if is_y else ax.xaxis

    tick_font_size = 12
    if getattr(axis, "tickfont", None) is not None and getattr(axis.tickfont, "size", None) is not None:
        tick_font_size = int(round(_font_px_to_pt(axis.tickfont.size, default=12.0)))
    ax.tick_params(axis=which, labelsize=tick_font_size)

    title_obj = getattr(axis, "title", None)
    title_text = _html_to_mpl_text(getattr(title_obj, "text", "")) if title_obj is not None else ""
    title_size = None
    if title_obj is not None and getattr(title_obj, "font", None) is not None:
        title_size = _font_px_to_pt(getattr(title_obj.font, "size", None), default=14.0)
    if is_y:
        ax.set_ylabel(title_text, fontsize=title_size)
    else:
        ax.set_xlabel(title_text, fontsize=title_size)

    if is_log:
        if is_y:
            ax.set_yscale("log")
        else:
            ax.set_xscale("log")

    data_range = _axis_range(axis, is_log=is_log)
    if data_range is not None:
        if is_y:
            ax.set_ylim(data_range)
        else:
            ax.set_xlim(data_range)

    show_grid = bool(getattr(axis, "showgrid", False))
    grid_color_raw = _parse_color(getattr(axis, "gridcolor", None), default=(0.75, 0.75, 0.75, 0.3))
    grid_color, major_grid_width = _soften_grid_style(grid_color_raw, 0.8, minor=False)
    if show_grid:
        ax.grid(True, axis=which, which="major", color=grid_color, linewidth=major_grid_width)
    else:
        ax.grid(False, axis=which, which="major")

    tickvals = _to_float_array(getattr(axis, "tickvals", None))
    ticktext = _to_string_list(getattr(axis, "ticktext", None))
    if tickvals.size and ticktext and tickvals.size == len(ticktext):
        locator = FixedLocator(tickvals.tolist())
        formatter = FixedFormatter([_html_to_mpl_text(text) for text in ticktext])
        mpl_axis.set_major_locator(locator)
        mpl_axis.set_major_formatter(formatter)
    else:
        dtick = getattr(axis, "dtick", None)
        locator = _make_multiple_locator(dtick, getattr(axis, "tick0", None)) if not is_log else None
        if locator is not None:
            mpl_axis.set_major_locator(locator)

        tickformat = getattr(axis, "tickformat", None)
        decimals_match = re.fullmatch(r"\.([0-9]+)f", str(tickformat)) if tickformat is not None else None
        if decimals_match:
            decimals = int(decimals_match.group(1))
            formatter = FuncFormatter(lambda value, _pos: f"{value:.{decimals}f}")
            mpl_axis.set_major_formatter(formatter)

    minor = getattr(axis, "minor", None)
    if minor is None:
        return

    minor_showgrid = bool(getattr(minor, "showgrid", False))
    minor_dtick = getattr(minor, "dtick", None)
    minor_tick0 = getattr(minor, "tick0", None)
    minor_locator = _make_multiple_locator(minor_dtick, minor_tick0) if not is_log else None
    if minor_locator is not None:
        mpl_axis.set_minor_locator(minor_locator)
    elif minor_showgrid and not is_log:
        if is_y:
            ax.minorticks_on()
        else:
            ax.minorticks_on()

    minor_grid_color_raw = _parse_color(getattr(minor, "gridcolor", None), default=(0.75, 0.75, 0.75, 0.16))
    minor_grid_color, minor_grid_width = _soften_grid_style(minor_grid_color_raw, 0.6, minor=True)
    if minor_showgrid:
        ax.grid(True, axis=which, which="minor", color=minor_grid_color, linewidth=minor_grid_width)
    else:
        ax.grid(False, axis=which, which="minor")


def _apply_layout(fig: go.Figure, mpl_figure: Figure, ax: Any) -> None:
    layout = fig.layout
    paper_bg = _parse_color(getattr(layout, "paper_bgcolor", None), default="white")
    plot_bg = _parse_color(getattr(layout, "plot_bgcolor", None), default="white")
    mpl_figure.patch.set_facecolor(paper_bg)
    ax.set_facecolor(plot_bg)
    ax.set_axisbelow(True)

    margin = layout.margin.to_plotly_json() if layout.margin is not None else {}
    width = float(layout.width) if layout.width is not None else 1000.0
    height = float(layout.height) if layout.height is not None else 700.0
    left = max(0.04, min(0.4, float(margin.get("l", 60.0)) / width))
    right = min(0.98, max(0.6, 1.0 - float(margin.get("r", 30.0)) / width))
    bottom = max(0.06, min(0.35, float(margin.get("b", 60.0)) / height))
    top = min(0.96, max(0.5, 1.0 - float(margin.get("t", 60.0)) / height))
    mpl_figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    if layout.font is not None and getattr(layout.font, "size", None) is not None:
        ax.tick_params(labelsize=int(round(_font_px_to_pt(layout.font.size, default=12.0) * 0.9)))

    _apply_axis_format(ax, layout.xaxis, which="x")
    _apply_axis_format(ax, layout.yaxis, which="y")

    title_obj = getattr(layout, "title", None)
    title_text = _html_to_mpl_text(getattr(title_obj, "text", "")) if title_obj is not None else ""
    if title_text:
        title_size = None
        if title_obj is not None and getattr(title_obj, "font", None) is not None:
            title_size = _font_px_to_pt(getattr(title_obj.font, "size", None), default=16.0)
        ax.set_title(title_text, fontsize=title_size, pad=18)


def _legend_label(trace: Any) -> str | None:
    if getattr(trace, "showlegend", None) is False:
        return None
    name = getattr(trace, "name", None)
    if name in (None, ""):
        return None
    return _html_to_mpl_text(name)


def _plot_scatter(ax: Any, trace: go.Scatter) -> None:
    x_vals = _to_float_array(getattr(trace, "x", None))
    y_vals = _to_float_array(getattr(trace, "y", None))
    count = min(x_vals.size, y_vals.size)
    if count == 0:
        return
    x_vals = x_vals[:count]
    y_vals = y_vals[:count]

    mode = str(getattr(trace, "mode", "markers")).lower()
    has_lines = "lines" in mode
    has_markers = "markers" in mode or mode == ""

    line = getattr(trace, "line", None)
    marker = getattr(trace, "marker", None)
    line_color = _parse_color(getattr(line, "color", None), default=_parse_color(getattr(marker, "color", None), default="black"))
    line_width = _linewidth_px_to_pt(getattr(line, "width", 1.5), default=1.5)
    line_style = _dash_to_mpl(getattr(line, "dash", None))

    marker_color = _parse_color(getattr(marker, "color", None), default=line_color)
    marker_edge = _parse_color(getattr(getattr(marker, "line", None), "color", None), default=line_color)
    marker_edge_width = _linewidth_px_to_pt(getattr(getattr(marker, "line", None), "width", 0.8), default=0.8)
    marker_size = _marker_size_to_points(getattr(marker, "size", None), default=7.0)
    alpha = float(getattr(trace, "opacity", 1.0) or 1.0)
    label = _legend_label(trace)

    y_error = _error_arrays(trace, "y", count)
    x_error = _error_arrays(trace, "x", count)
    if has_lines:
        ax.plot(
            x_vals,
            y_vals,
            color=line_color,
            linewidth=line_width,
            linestyle=line_style,
            alpha=alpha,
            label=label if not has_markers else None,
            zorder=2.6,
        )
    if has_markers:
        ax.scatter(
            x_vals,
            y_vals,
            s=max(6.0, marker_size**2 / 1.6),
            c=[marker_color],
            edgecolors=[marker_edge],
            linewidths=marker_edge_width,
            alpha=alpha,
            label=label,
            zorder=3.2,
        )
    if x_error is not None or y_error is not None:
        y_minus, y_plus = y_error if y_error is not None else (None, None)
        x_minus, x_plus = x_error if x_error is not None else (None, None)
        y_err_cfg = getattr(trace, "error_y", None)
        x_err_cfg = getattr(trace, "error_x", None)
        err_cfg = y_err_cfg if y_err_cfg is not None else x_err_cfg
        ecolor = _parse_color(
            getattr(y_err_cfg, "color", None),
            default=_parse_color(getattr(x_err_cfg, "color", None), default="black"),
        )
        elinewidth = _linewidth_px_to_pt(getattr(err_cfg, "thickness", 1.2), default=1.2)
        capsize = max(0.0, _linewidth_px_to_pt(getattr(err_cfg, "width", 0.0), default=0.0) * 0.45)
        ax.errorbar(
            x_vals,
            y_vals,
            xerr=None if x_minus is None or x_plus is None else np.vstack([x_minus, x_plus]),
            yerr=None if y_minus is None or y_plus is None else np.vstack([y_minus, y_plus]),
            fmt="none",
            ecolor=ecolor,
            elinewidth=elinewidth,
            capsize=capsize,
            alpha=alpha,
            zorder=2.2,
        )


def _plot_bar(ax: Any, trace: go.Bar) -> None:
    x_vals = _to_float_array(getattr(trace, "x", None))
    y_vals = _to_float_array(getattr(trace, "y", None))
    count = min(x_vals.size, y_vals.size)
    if count == 0:
        return
    x_vals = x_vals[:count]
    y_vals = y_vals[:count]
    width_vals = _to_float_array(getattr(trace, "width", None))
    if width_vals.size == 0:
        width = 0.8
    elif width_vals.size == 1:
        width = float(width_vals[0])
    else:
        width = width_vals[:count]
    marker = getattr(trace, "marker", None)
    face = _parse_color(getattr(marker, "color", None), default="#7aa6ff")
    edge = _parse_color(getattr(getattr(marker, "line", None), "color", None), default=face)
    edge_width = _linewidth_px_to_pt(getattr(getattr(marker, "line", None), "width", 1.0), default=1.0)
    alpha = float(getattr(trace, "opacity", 1.0) or 1.0)
    ax.bar(
        x_vals,
        y_vals,
        width=width,
        color=face,
        edgecolor=edge,
        linewidth=edge_width,
        alpha=alpha,
        align="center",
        label=_legend_label(trace),
        zorder=2.0,
    )


def _draw_shape(ax: Any, shape: Any) -> None:
    shape_type = str(getattr(shape, "type", "")).lower()
    if shape_type not in {"line", "rect", "circle"}:
        return

    xref = str(getattr(shape, "xref", "x"))
    yref = str(getattr(shape, "yref", "y"))
    if xref == "paper" and yref == "paper":
        transform = ax.transAxes
    elif xref == "paper":
        transform = blended_transform_factory(ax.transAxes, ax.transData)
    elif yref == "paper":
        transform = blended_transform_factory(ax.transData, ax.transAxes)
    else:
        transform = ax.transData

    line = getattr(shape, "line", None)
    color_raw = _parse_color(getattr(line, "color", None), default=(0.5, 0.5, 0.5, 0.3))
    width_raw = _linewidth_px_to_pt(getattr(line, "width", 1.0), default=1.0)
    if _is_gridline_shape(shape, color_raw):
        color, width = _soften_grid_style(color_raw, width_raw, minor=(width_raw <= 0.8))
    else:
        color = color_raw
        width = width_raw
    dash = _dash_to_mpl(getattr(line, "dash", None))
    fillcolor = _parse_color(getattr(shape, "fillcolor", None), default=(0, 0, 0, 0))

    if shape_type == "line":
        artist = Line2D(
            [float(shape.x0), float(shape.x1)],
            [float(shape.y0), float(shape.y1)],
            transform=transform,
            color=color,
            linewidth=width,
            linestyle=dash,
            zorder=0.4,
        )
        ax.add_line(artist)
        return

    x0 = float(shape.x0)
    x1 = float(shape.x1)
    y0 = float(shape.y0)
    y1 = float(shape.y1)
    left = min(x0, x1)
    bottom = min(y0, y1)
    width_value = abs(x1 - x0)
    height_value = abs(y1 - y0)

    if shape_type == "rect":
        patch = Rectangle(
            (left, bottom),
            width_value,
            height_value,
            transform=transform,
            facecolor=fillcolor,
            edgecolor=color,
            linewidth=width,
            linestyle=dash,
            zorder=0.35,
        )
    else:
        patch = Ellipse(
            (left + (width_value / 2.0), bottom + (height_value / 2.0)),
            width=width_value,
            height=height_value,
            transform=transform,
            facecolor=fillcolor,
            edgecolor=color,
            linewidth=width,
            linestyle=dash,
            zorder=0.35,
        )
    ax.add_patch(patch)


def _draw_annotation(mpl_figure: Figure, ax: Any, annotation: Any) -> None:
    text = _html_to_mpl_text(getattr(annotation, "text", ""))
    if not text.strip():
        return

    xref = str(getattr(annotation, "xref", "x"))
    yref = str(getattr(annotation, "yref", "y"))
    if xref == "paper" and yref == "paper":
        transform = ax.transAxes
    elif xref == "paper":
        transform = blended_transform_factory(ax.transAxes, ax.transData)
    elif yref == "paper":
        transform = blended_transform_factory(ax.transData, ax.transAxes)
    else:
        transform = ax.transData

    xshift = float(getattr(annotation, "xshift", 0.0) or 0.0) * 0.75
    yshift = float(getattr(annotation, "yshift", 0.0) or 0.0) * 0.75
    if xshift or yshift:
        transform = offset_copy(transform, fig=mpl_figure, x=xshift, y=yshift, units="points")

    font = getattr(annotation, "font", None)
    font_size = int(round(_font_px_to_pt(getattr(font, "size", 12), default=12.0)))
    font_color = _parse_color(getattr(font, "color", None), default="black")

    bbox = None
    bgcolor = getattr(annotation, "bgcolor", None)
    bordercolor = getattr(annotation, "bordercolor", None)
    borderwidth = _linewidth_px_to_pt(getattr(annotation, "borderwidth", 0.0), default=0.0)
    if bgcolor is not None or bordercolor is not None or borderwidth > 0:
        bbox = {
            "boxstyle": "round,pad=0.3",
            "facecolor": _parse_color(bgcolor, default=(1, 1, 1, 0)),
            "edgecolor": _parse_color(bordercolor, default=(0, 0, 0, 0)),
            "linewidth": borderwidth,
        }

    ha = {
        "left": "left",
        "right": "right",
        "center": "center",
    }.get(str(getattr(annotation, "xanchor", "center")), "center")
    va = {
        "top": "top",
        "bottom": "bottom",
        "middle": "center",
        "center": "center",
    }.get(str(getattr(annotation, "yanchor", "middle")), "center")

    ax.text(
        float(getattr(annotation, "x", 0.0)),
        float(getattr(annotation, "y", 0.0)),
        text,
        transform=transform,
        fontsize=font_size,
        color=font_color,
        ha=ha,
        va=va,
        multialignment=str(getattr(annotation, "align", "left") or "left"),
        bbox=bbox,
        clip_on=False,
        zorder=5.0,
    )


def _draw_traces(ax: Any, fig: go.Figure) -> int:
    rendered = 0
    for trace in fig.data:
        if not _is_visible_trace(trace):
            continue
        trace_type = str(getattr(trace, "type", "scatter")).lower()
        if trace_type == "scatter":
            _plot_scatter(ax, trace)
            rendered += 1
        elif trace_type == "bar":
            _plot_bar(ax, trace)
            rendered += 1
    return rendered


def _draw_shapes(ax: Any, fig: go.Figure) -> int:
    rendered = 0
    for shape in fig.layout.shapes or []:
        before_lines = len(ax.lines)
        before_patches = len(ax.patches)
        _draw_shape(ax, shape)
        if len(ax.lines) > before_lines or len(ax.patches) > before_patches:
            rendered += 1
    return rendered


def _draw_annotations(mpl_figure: Figure, ax: Any, fig: go.Figure) -> int:
    rendered = 0
    for annotation in fig.layout.annotations or []:
        before_texts = len(ax.texts)
        _draw_annotation(mpl_figure, ax, annotation)
        if len(ax.texts) > before_texts:
            rendered += 1
    return rendered


def _draw_legend(ax: Any, fig: go.Figure) -> bool:
    legend = getattr(fig.layout, "legend", None)
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(handle, label) for handle, label in zip(handles, labels, strict=True) if label and label != "_nolegend_"]
    if not filtered:
        return False

    handles, labels = zip(*filtered, strict=True)
    font_size = _font_px_to_pt(getattr(getattr(legend, "font", None), "size", None), default=11.0) if legend is not None else 11.0
    bbox_face = _parse_color(getattr(legend, "bgcolor", None), default=(1, 1, 1, 0.88)) if legend is not None else (1, 1, 1, 0.88)
    orientation = str(getattr(legend, "orientation", "v")).lower() if legend is not None else "v"

    figure = ax.figure
    axes_bbox = ax.get_position()

    if orientation == "h":
        ncols = min(len(labels), 4)
        rows = int(np.ceil(len(labels) / max(1, ncols)))
        fig_height_in = max(1.0, figure.get_figheight())
        row_height = max(0.026, (font_size / (72.0 * fig_height_in)) * 1.9)
        top_margin_needed = min(0.30, max(0.085, (rows * row_height) + 0.028))
        current_top_margin = max(0.0, 1.0 - axes_bbox.y1)
        if current_top_margin < top_margin_needed:
            new_top = max(0.42, 1.0 - top_margin_needed)
            figure.subplots_adjust(top=new_top)
            axes_bbox = ax.get_position()
            current_top_margin = max(0.0, 1.0 - axes_bbox.y1)

        box_left = axes_bbox.x0
        box_bottom = axes_bbox.y1 + max(0.006, current_top_margin * 0.08)
        box_height = max(0.04, current_top_margin * 0.84)
        box_width = max(0.18, axes_bbox.width)

        ax.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(box_left, box_bottom, box_width, box_height),
            bbox_transform=figure.transFigure,
            mode="expand",
            ncol=ncols,
            frameon=True,
            facecolor=bbox_face,
            edgecolor=(0, 0, 0, 0.18),
            fontsize=font_size,
            columnspacing=1.2,
            handlelength=2.4,
            borderaxespad=0.0,
        )
        return True

    x_anchor = str(getattr(legend, "xanchor", "left")).lower() if legend is not None else "left"
    y_anchor = str(getattr(legend, "yanchor", "top")).lower() if legend is not None else "top"
    x = float(getattr(legend, "x", 0.0) if legend is not None and getattr(legend, "x", None) is not None else 0.0)
    y = float(getattr(legend, "y", 1.0) if legend is not None and getattr(legend, "y", None) is not None else 1.0)
    if x_anchor == "auto":
        x_anchor = "left" if x <= 0.33 else ("center" if x < 0.66 else "right")
    if y_anchor == "auto":
        y_anchor = "bottom" if y >= 0.5 else "top"

    x_loc = {"left": "left", "center": "center", "right": "right"}.get(x_anchor, "left")
    y_loc = {"top": "upper", "middle": "center", "center": "center", "bottom": "lower"}.get(y_anchor, "upper")
    loc = f"{y_loc} {x_loc}".strip()
    loc = {"center left": "center left", "center center": "center", "center right": "center right"}.get(loc, loc)

    clamped_x = float(np.clip(x, 0.02, 0.98))
    clamped_y = float(np.clip(y, 0.02, 0.98))
    ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=(clamped_x, clamped_y),
        bbox_transform=figure.transFigure,
        ncol=1,
        frameon=True,
        facecolor=bbox_face,
        edgecolor=(0, 0, 0, 0.18),
        fontsize=font_size,
        columnspacing=1.2,
        handlelength=2.4,
        borderaxespad=0.0,
    )
    return True


def _ensure_export_decorations_fit(mpl_figure: Figure, ax: Any) -> None:
    """Expand subplot margins so axis labels, ticks, and legend stay inside the canvas."""
    canvas = FigureCanvasAgg(mpl_figure)
    pad = 0.012

    for _ in range(3):
        canvas.draw()
        renderer = canvas.get_renderer()
        bboxes: list[Bbox] = []

        def _append_artist_bbox(artist: Any) -> None:
            if artist is None:
                return
            if hasattr(artist, "get_visible") and not artist.get_visible():
                return
            if hasattr(artist, "get_text") and not str(artist.get_text() or "").strip():
                return
            try:
                bbox = artist.get_window_extent(renderer=renderer)
            except Exception:
                return
            if bbox.width > 0 and bbox.height > 0:
                bboxes.append(bbox)

        _append_artist_bbox(ax.xaxis.label)
        _append_artist_bbox(ax.yaxis.label)
        _append_artist_bbox(ax.title)
        _append_artist_bbox(ax.xaxis.get_offset_text())
        _append_artist_bbox(ax.yaxis.get_offset_text())
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            _append_artist_bbox(label)

        legend = ax.get_legend()
        if legend is not None:
            _append_artist_bbox(legend)

        if not bboxes:
            return

        union = Bbox.union(bboxes).transformed(mpl_figure.transFigure.inverted())
        left_over = max(0.0, pad - float(union.x0))
        right_over = max(0.0, float(union.x1) - (1.0 - pad))
        bottom_over = max(0.0, pad - float(union.y0))
        top_over = max(0.0, float(union.y1) - (1.0 - pad))

        if max(left_over, right_over, bottom_over, top_over) <= 1e-4:
            return

        pos = ax.get_position()
        new_left = min(0.42, float(pos.x0) + left_over)
        new_right = max(new_left + 0.25, float(pos.x1) - right_over)
        new_bottom = min(0.42, float(pos.y0) + bottom_over)
        new_top = max(new_bottom + 0.25, float(pos.y1) - top_over)
        mpl_figure.subplots_adjust(left=new_left, right=min(0.98, new_right), bottom=new_bottom, top=min(0.98, new_top))


def render_plotly_figure_to_matplotlib(
    fig: go.Figure,
    *,
    width: int | None = None,
    height: int | None = None,
    base_dpi: int = 300,
) -> tuple[Figure, Any, ExportRenderSummary]:
    """Render a Plotly figure to Matplotlib and return a parity summary for tests."""
    validate_supported_export_features(fig)

    width_px = int(width or (fig.layout.width if fig.layout.width is not None else 1200))
    height_px = int(height or (fig.layout.height if fig.layout.height is not None else 800))
    dpi = max(72, int(base_dpi))
    figsize = (max(1.0, width_px / dpi), max(1.0, height_px / dpi))

    mpl_figure = Figure(figsize=figsize, dpi=dpi)
    ax = mpl_figure.add_subplot(111)

    _apply_layout(fig, mpl_figure, ax)
    rendered_shapes = _draw_shapes(ax, fig)
    rendered_traces = _draw_traces(ax, fig)
    rendered_annotations = _draw_annotations(mpl_figure, ax, fig)
    legend_present = _draw_legend(ax, fig)
    _ensure_export_decorations_fit(mpl_figure, ax)

    summary = ExportRenderSummary(
        expected_trace_count=sum(1 for trace in fig.data if _is_visible_trace(trace) and str(getattr(trace, "type", "scatter")).lower() in SUPPORTED_TRACE_TYPES),
        rendered_trace_count=rendered_traces,
        expected_shape_count=len(_visible_supported_shapes(fig)),
        rendered_shape_count=rendered_shapes,
        expected_annotation_count=len(_visible_annotations(fig)),
        rendered_annotation_count=rendered_annotations,
        legend_present=legend_present,
        xaxis_title=ax.get_xlabel(),
        yaxis_title=ax.get_ylabel(),
        plot_title=ax.get_title(),
        xscale=ax.get_xscale(),
        yscale=ax.get_yscale(),
    )
    return mpl_figure, ax, summary


def plotly_figure_to_image_bytes(
    fig: go.Figure,
    fmt: str,
    *,
    width: int | None = None,
    height: int | None = None,
    scale: float = 1.0,
    base_dpi: int = 300,
) -> bytes:
    """Render a Plotly figure via Matplotlib for stable local exports."""
    if fmt not in {"png", "svg", "pdf"}:
        raise ValueError("Supported formats are 'png', 'svg', and 'pdf'.")

    dpi = max(72, int(base_dpi))
    scale_value = max(0.2, float(scale))
    mpl_figure, _ax, _summary = render_plotly_figure_to_matplotlib(
        fig,
        width=width,
        height=height,
        base_dpi=dpi,
    )

    output = BytesIO()
    save_kwargs: dict[str, Any] = {
        "format": fmt,
        "facecolor": mpl_figure.get_facecolor(),
        "bbox_inches": None,
    }
    if fmt == "png":
        save_kwargs["dpi"] = dpi * scale_value
    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}):
        mpl_figure.savefig(output, **save_kwargs)
    output.seek(0)
    data = output.getvalue()
    mpl_figure.clear()
    return data
