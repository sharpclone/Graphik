"""Figure export and plot-annotation helpers."""

from __future__ import annotations

from datetime import datetime
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import SUMMARY_FLOAT_PRECISION
from .i18n import translate


PLOT_INFO_ANNOTATION_NAME = "_plot_info_block"


def add_plot_text_block(fig: go.Figure, lines: list[str], font_size: int = 12) -> None:
    """Add equation block inside plot area, auto-placed in a low-overlap zone."""
    if not lines:
        return

    raw_lines = [str(line).strip() for line in lines if str(line).strip()]
    if not raw_lines:
        return

    def _wrap_lines(source_lines: list[str], width_chars: int) -> list[str]:
        wrapped_lines: list[str] = []
        for raw_line in source_lines:
            wrapped = textwrap.wrap(
                raw_line,
                width=width_chars,
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped_lines.extend(wrapped if wrapped else [raw_line])
        return wrapped_lines

    def _to_float_array(values: object) -> np.ndarray:
        if values is None:
            return np.array([], dtype=float)
        try:
            arr = np.asarray(values, dtype=float)
            if arr.ndim == 0:
                return np.array([float(arr)], dtype=float)
            return arr.astype(float)
        except (TypeError, ValueError):
            try:
                return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
            except Exception:
                return np.array([], dtype=float)

    def _array_to_length(values: object, length: int, fill_value: float = 0.0) -> np.ndarray:
        if length <= 0:
            return np.array([], dtype=float)
        arr = _to_float_array(values)
        if arr.size == 0:
            return np.full(length, float(fill_value), dtype=float)
        if arr.size == 1 and length > 1:
            return np.full(length, float(arr[0]), dtype=float)
        if arr.size < length:
            out = np.full(length, float(fill_value), dtype=float)
            out[: arr.size] = arr
            return out
        return arr[:length]

    def _trace_xy_samples(trace: go.Scatter) -> tuple[np.ndarray, np.ndarray]:
        x_raw = _to_float_array(getattr(trace, "x", None))
        y_raw = _to_float_array(getattr(trace, "y", None))
        n = min(x_raw.size, y_raw.size)
        if n == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        x_vals = x_raw[:n]
        y_vals = y_raw[:n]
        finite = np.isfinite(x_vals) & np.isfinite(y_vals)
        return x_vals[finite], y_vals[finite]

    def _extract_axis_bounds() -> tuple[float, float, float, float, bool] | None:
        x_axis = fig.layout.xaxis
        y_axis = fig.layout.yaxis
        y_is_log = str(getattr(y_axis, "type", "linear")).lower() == "log"

        x_range = getattr(x_axis, "range", None)
        if x_range is not None and len(x_range) == 2 and x_range[0] is not None and x_range[1] is not None:
            x0, x1 = float(x_range[0]), float(x_range[1])
        else:
            x_samples: list[float] = []
            for trace in fig.data:
                if getattr(trace, "visible", True) in (False, "legendonly"):
                    continue
                x_vals, _ = _trace_xy_samples(trace)
                if x_vals.size:
                    x_samples.extend(x_vals.tolist())
            if not x_samples:
                return None
            x0, x1 = float(np.min(x_samples)), float(np.max(x_samples))

        y_range = getattr(y_axis, "range", None)
        if y_range is not None and len(y_range) == 2 and y_range[0] is not None and y_range[1] is not None:
            y0_raw, y1_raw = float(y_range[0]), float(y_range[1])
            if y_is_log:
                y0, y1 = float(10.0**y0_raw), float(10.0**y1_raw)
            else:
                y0, y1 = y0_raw, y1_raw
        else:
            y_samples: list[float] = []
            for trace in fig.data:
                if getattr(trace, "visible", True) in (False, "legendonly"):
                    continue
                _, y_vals = _trace_xy_samples(trace)
                if y_vals.size:
                    if y_is_log:
                        y_vals = y_vals[y_vals > 0]
                    if y_vals.size:
                        y_samples.extend(y_vals.tolist())
            if not y_samples:
                return None
            y0, y1 = float(np.min(y_samples)), float(np.max(y_samples))

        x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
        y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)
        if x_max <= x_min:
            x_pad = max(1.0, abs(x_min) * 0.05)
            x_min -= x_pad
            x_max += x_pad
        if y_max <= y_min:
            y_pad = max(1.0, abs(y_min) * 0.05)
            y_min -= y_pad
            y_max += y_pad
        if y_is_log and (y_min <= 0 or y_max <= 0):
            return None
        return x_min, x_max, y_min, y_max, y_is_log

    def _collect_normalized_points(
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        y_is_log: bool,
    ) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        x_span = max(1e-12, x_max - x_min)
        if y_is_log:
            y_min_log = np.log10(y_min)
            y_max_log = np.log10(y_max)
            y_span_log = max(1e-12, y_max_log - y_min_log)
        else:
            y_span = max(1e-12, y_max - y_min)

        for trace in fig.data:
            if getattr(trace, "visible", True) in (False, "legendonly"):
                continue
            x_vals, y_vals = _trace_xy_samples(trace)
            n = min(x_vals.size, y_vals.size)
            if n == 0:
                continue

            if n > 250:
                idx = np.linspace(0, n - 1, 250, dtype=int)
                x_vals = x_vals[idx]
                y_vals = y_vals[idx]
                n = 250

            if y_is_log:
                mask = y_vals > 0
                x_vals = x_vals[mask]
                y_vals = y_vals[mask]
                if y_vals.size == 0:
                    continue
                y_norm = (np.log10(y_vals) - y_min_log) / y_span_log
            else:
                y_norm = (y_vals - y_min) / y_span
            x_norm = (x_vals - x_min) / x_span

            valid = np.isfinite(x_norm) & np.isfinite(y_norm)
            xn_valid = x_norm[valid]
            yn_valid = y_norm[valid]
            for xn, yn in zip(xn_valid, yn_valid):
                if -0.05 <= xn <= 1.05 and -0.05 <= yn <= 1.05:
                    points.append((float(xn), float(yn)))

            mode_value = str(getattr(trace, "mode", "")).lower()
            if "lines" in mode_value and xn_valid.size >= 2:
                for idx_seg in range(xn_valid.size - 1):
                    x0_seg = float(xn_valid[idx_seg])
                    x1_seg = float(xn_valid[idx_seg + 1])
                    y0_seg = float(yn_valid[idx_seg])
                    y1_seg = float(yn_valid[idx_seg + 1])
                    seg_len = float(np.hypot(x1_seg - x0_seg, y1_seg - y0_seg))
                    steps = int(np.clip(np.ceil(seg_len * 160.0), 6, 80))
                    t_values = np.linspace(0.0, 1.0, steps)
                    for t in t_values:
                        xn = x0_seg + (x1_seg - x0_seg) * float(t)
                        yn = y0_seg + (y1_seg - y0_seg) * float(t)
                        if -0.05 <= xn <= 1.05 and -0.05 <= yn <= 1.05:
                            points.append((xn, yn))

            err = getattr(trace, "error_y", None)
            if err is None or not bool(getattr(err, "visible", False)):
                continue
            plus = _array_to_length(getattr(err, "array", None), n, fill_value=0.0)
            minus_src = getattr(err, "arrayminus", None)
            minus = _array_to_length(minus_src, n, fill_value=np.nan)
            if np.isnan(minus).all():
                minus = plus.copy()
            y_up = y_vals[:n] + plus[:n]
            y_dn = y_vals[:n] - minus[:n]

            if y_is_log:
                for y_extra in (y_up, y_dn):
                    mask = y_extra > 0
                    if not np.any(mask):
                        continue
                    x_e = x_vals[:n][mask]
                    y_e = y_extra[mask]
                    x_n = (x_e - x_min) / x_span
                    y_n = (np.log10(y_e) - y_min_log) / y_span_log
                    valid_e = np.isfinite(x_n) & np.isfinite(y_n)
                    for xn, yn in zip(x_n[valid_e], y_n[valid_e]):
                        if -0.05 <= xn <= 1.05 and -0.05 <= yn <= 1.05:
                            points.append((float(xn), float(yn)))
            else:
                for y_extra in (y_up, y_dn):
                    x_n = (x_vals[:n] - x_min) / x_span
                    y_n = (y_extra - y_min) / y_span
                    valid_e = np.isfinite(x_n) & np.isfinite(y_n)
                    for xn, yn in zip(x_n[valid_e], y_n[valid_e]):
                        if -0.05 <= xn <= 1.05 and -0.05 <= yn <= 1.05:
                            points.append((float(xn), float(yn)))

        return points

    effective_font_size = float(max(10, int(font_size)))
    max_box_w = 0.92
    max_box_h = 0.86

    margin_obj = fig.layout.margin.to_plotly_json() if fig.layout.margin is not None else {}
    fig_w_px = float(fig.layout.width) if fig.layout.width is not None else 1200.0
    fig_h_px = float(fig.layout.height) if fig.layout.height is not None else 700.0
    margin_l = float(margin_obj.get("l", 60))
    margin_r = float(margin_obj.get("r", 30))
    margin_t = float(margin_obj.get("t", 60))
    margin_b = float(margin_obj.get("b", 60))
    plot_w_px = max(220.0, fig_w_px - margin_l - margin_r)
    plot_h_px = max(180.0, fig_h_px - margin_t - margin_b)

    def _estimate_box_size(lines_for_box: list[str], current_font_size: float) -> tuple[float, float]:
        max_line_len = max(len(line) for line in lines_for_box) if lines_for_box else 20
        line_h_px = max(10.0, 1.20 * float(current_font_size))
        char_w_px = max(4.0, 0.52 * float(current_font_size))
        est_w_px = max(90.0, 18.0 + char_w_px * float(max_line_len))
        est_h_px = max(40.0, 12.0 + line_h_px * float(len(lines_for_box)))
        return (est_w_px / plot_w_px, est_h_px / plot_h_px)

    def _preferred_wrap_width(current_font_size: float) -> int:
        char_w_px = max(4.0, 0.52 * float(current_font_size))
        usable_w_px = max(140.0, (max_box_w * plot_w_px) - 20.0)
        return int(np.clip(np.floor(usable_w_px / char_w_px), 24, 260))

    display_lines = list(raw_lines)
    wrap_width = _preferred_wrap_width(effective_font_size)
    if any(len(line) > wrap_width for line in raw_lines):
        display_lines = _wrap_lines(raw_lines, wrap_width)

    box_w, box_h = _estimate_box_size(display_lines, effective_font_size)

    for _ in range(14):
        if box_w <= max_box_w:
            break
        next_wrap = max(20, int(round(wrap_width * 0.93)))
        if next_wrap >= wrap_width:
            break
        wrap_width = next_wrap
        display_lines = _wrap_lines(raw_lines, wrap_width)
        box_w, box_h = _estimate_box_size(display_lines, effective_font_size)

    for _ in range(10):
        if box_h <= max_box_h:
            break
        next_wrap = min(260, int(round(wrap_width * 1.10)))
        if next_wrap <= wrap_width:
            break
        wrap_width = next_wrap
        if any(len(line) > wrap_width for line in raw_lines):
            display_lines = _wrap_lines(raw_lines, wrap_width)
        else:
            display_lines = list(raw_lines)
        box_w, box_h = _estimate_box_size(display_lines, effective_font_size)

    soft_min_font_size = max(11.0, float(font_size) * 0.90)
    hard_min_font_size = max(10.0, float(font_size) * 0.78)
    for _ in range(10):
        if box_w <= max_box_w and box_h <= max_box_h:
            break
        next_font_size = max(soft_min_font_size, effective_font_size * 0.96)
        if abs(next_font_size - effective_font_size) < 1e-9:
            break
        effective_font_size = next_font_size
        wrap_width = _preferred_wrap_width(effective_font_size)
        if any(len(line) > wrap_width for line in raw_lines):
            display_lines = _wrap_lines(raw_lines, wrap_width)
        else:
            display_lines = list(raw_lines)
        box_w, box_h = _estimate_box_size(display_lines, effective_font_size)

    for _ in range(10):
        if box_w <= max_box_w and box_h <= max_box_h:
            break
        next_font_size = max(hard_min_font_size, effective_font_size * 0.92)
        if abs(next_font_size - effective_font_size) < 1e-9:
            break
        effective_font_size = next_font_size
        wrap_width = _preferred_wrap_width(effective_font_size)
        if any(len(line) > wrap_width for line in raw_lines):
            display_lines = _wrap_lines(raw_lines, wrap_width)
        else:
            display_lines = list(raw_lines)
        box_w, box_h = _estimate_box_size(display_lines, effective_font_size)

    max_grow_font = max(float(font_size), float(font_size) * 1.35)
    for _ in range(14):
        if box_w >= (0.78 * max_box_w) or box_h >= (0.78 * max_box_h):
            break
        next_font_size = min(max_grow_font, effective_font_size * 1.06)
        if next_font_size <= effective_font_size + 1e-9:
            break
        candidate_wrap = _preferred_wrap_width(next_font_size)
        if any(len(line) > candidate_wrap for line in raw_lines):
            candidate_lines = _wrap_lines(raw_lines, candidate_wrap)
        else:
            candidate_lines = list(raw_lines)
        cand_w, cand_h = _estimate_box_size(candidate_lines, next_font_size)
        if cand_w > max_box_w or cand_h > max_box_h:
            break
        effective_font_size = next_font_size
        wrap_width = candidate_wrap
        display_lines = candidate_lines
        box_w, box_h = cand_w, cand_h

    box_w = float(min(box_w, max_box_w))
    box_h = float(min(box_h, max_box_h))

    bounds = _extract_axis_bounds()
    x_pos = 0.02
    y_pos = 0.98
    if bounds is not None:
        x_min, x_max, y_min, y_max, y_is_log = bounds
        points = _collect_normalized_points(x_min, x_max, y_min, y_max, y_is_log)

        x_margin = 0.02
        y_margin = 0.03
        # Reserve extra space above the bottom axis so the box stays clear
        # in smaller responsive previews where wrapped text makes it visually taller.
        bottom_axis_clearance = float(np.clip(0.055 + (0.18 * box_h), 0.08, 0.18))
        x_left_max = max(x_margin, 1.0 - x_margin - box_w)
        y_top_min = box_h + y_margin + bottom_axis_clearance
        y_top_max = 1.0 - y_margin

        x_grid_count = 7
        y_grid_count = 8
        x_positions = np.linspace(x_margin, x_left_max, x_grid_count).tolist()
        y_positions = np.linspace(y_top_max, y_top_min, y_grid_count).tolist()

        base_candidates: list[tuple[float, float]] = [
            (x_margin, y_top_max),
            (x_left_max, y_top_max),
            (x_margin, 0.75),
            (x_left_max, 0.75),
            (x_margin, 0.55),
            (x_left_max, 0.55),
        ]
        for top in y_positions:
            for left in x_positions:
                base_candidates.append((float(left), float(top)))

        candidates: list[tuple[float, float, float, float, float, float]] = []
        for left_raw, top_raw in base_candidates:
            left = float(np.clip(left_raw, x_margin, x_left_max))
            top = float(np.clip(top_raw, y_top_min, y_top_max))
            right = left + box_w
            bottom = top - box_h
            candidates.append((left, top, right, bottom, left_raw, top_raw))

        def _candidate_key(candidate: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float]:
            left, top, right, bottom, left_raw, _ = candidate
            pad = 0.025
            overlap_count = 0
            min_dist = 1.0
            for xn, yn in points:
                if (left - pad) <= xn <= (right + pad) and (bottom - pad) <= yn <= (top + pad):
                    overlap_count += 1
                dx = 0.0
                if xn < left:
                    dx = left - xn
                elif xn > right:
                    dx = xn - right
                dy = 0.0
                if yn < bottom:
                    dy = bottom - yn
                elif yn > top:
                    dy = yn - top
                dist = float(np.hypot(dx, dy))
                if dist < min_dist:
                    min_dist = dist
            bottom_penalty = max(0.0, (bottom_axis_clearance + y_margin) - bottom)
            # Prefer lower overlap first, then greater distance from data,
            # then more clearance from the bottom/x-axis, then higher positions.
            return (
                float(overlap_count),
                float(-min_dist),
                float(bottom_penalty),
                float(-top),
            )

        best = min(candidates, key=_candidate_key)
        x_pos, y_pos = best[0], best[1]

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=x_pos,
        y=y_pos,
        text="<br>".join(display_lines),
        showarrow=False,
        align="left",
        xanchor="left",
        yanchor="top",
        font={"size": int(round(effective_font_size))},
        bgcolor="rgba(255,255,255,0.72)",
        bordercolor="rgba(0,0,0,0.22)",
        borderwidth=1,
        name=PLOT_INFO_ANNOTATION_NAME,
    )


def remove_plot_text_block(fig: go.Figure, lines: list[str]) -> None:
    """Remove previously inserted equation text block matching current content."""
    annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    target_text = "<br>".join(str(line).strip() for line in lines if str(line).strip()) if lines else ""
    filtered = []
    for ann in annotations:
        ann_name = getattr(ann, "name", None)
        ann_template_name = getattr(ann, "templateitemname", None)
        ann_hover = getattr(ann, "hovertext", None)
        ann_text = getattr(ann, "text", None)
        is_plot_info = (
            ann_name == PLOT_INFO_ANNOTATION_NAME
            or ann_template_name == PLOT_INFO_ANNOTATION_NAME
            or ann_hover == PLOT_INFO_ANNOTATION_NAME
            or (target_text and ann_text == target_text)
        )
        if not is_plot_info:
            filtered.append(ann)
    if len(filtered) != len(annotations):
        fig.layout.annotations = filtered


def place_plot_text_block(fig: go.Figure, lines: list[str], font_size: int) -> None:
    """Reposition equation block by removing old one and placing with current layout."""
    remove_plot_text_block(fig, lines)
    add_plot_text_block(fig, lines, font_size=font_size)


def paper_size_mm(name: str) -> tuple[float, float]:
    """Return paper size in mm (width, height) for portrait orientation."""
    paper_sizes = {
        "A4": (210.0, 297.0),
        "A5": (148.0, 210.0),
        "Letter": (215.9, 279.4),
    }
    return paper_sizes.get(name, paper_sizes["A4"])


def mm_to_px(mm: float, dpi: int) -> int:
    """Convert millimeters to pixels at target DPI."""
    return int(round((mm / 25.4) * float(dpi)))


def scale_figure_for_export(
    fig: go.Figure,
    visual_scale: float,
    target_text_pt: float | None = None,
    base_export_dpi: int = 300,
) -> go.Figure:
    """Scale fonts, markers, and line widths for export readability."""
    out = go.Figure(fig)
    original_base_font = (
        float(out.layout.font.size) if out.layout.font and out.layout.font.size is not None else 12.0
    )

    auto_factor = 1.0
    if target_text_pt is not None:
        target_px = (float(target_text_pt) / 72.0) * float(base_export_dpi)
        auto_factor = max(0.1, target_px / max(1.0, original_base_font))

    factor = max(0.2, auto_factor * float(visual_scale))
    if abs(factor - 1.0) < 1e-9:
        return out

    def _scaled(value: float | int | None, default: float) -> float:
        base = default if value is None else float(value)
        return max(1.0, base * factor)

    if out.layout.font is None:
        out.layout.font = {}
    out.layout.font.size = _scaled(out.layout.font.size, original_base_font)

    if out.layout.legend is not None:
        if out.layout.legend.font is None:
            out.layout.legend.font = {}
        legend_default = original_base_font * 0.9
        out.layout.legend.font.size = _scaled(out.layout.legend.font.size, legend_default)

    for axis_name in ("xaxis", "yaxis"):
        axis = getattr(out.layout, axis_name, None)
        if axis is None:
            continue
        if axis.title is not None:
            if axis.title.font is None:
                axis.title.font = {}
            axis.title.font.size = _scaled(axis.title.font.size, original_base_font * 1.1)
        if axis.tickfont is None:
            axis.tickfont = {}
        axis.tickfont.size = _scaled(axis.tickfont.size, original_base_font * 0.9)

    for ann in out.layout.annotations or []:
        if ann.font is None:
            ann.font = {}
        ann.font.size = _scaled(ann.font.size, original_base_font * 0.9)

    marker_factor = max(0.6, factor**0.8)
    stroke_factor = max(0.6, factor)
    error_factor = max(0.8, factor * 1.15)

    for trace in out.data:
        if hasattr(trace, "line") and trace.line is not None:
            base_width = 1.5 if trace.line.width is None else float(trace.line.width)
            trace.line.width = max(0.8, base_width * stroke_factor)
        if hasattr(trace, "marker") and trace.marker is not None:
            base_size = 7.0 if trace.marker.size is None else float(trace.marker.size)
            trace.marker.size = max(1.5, base_size * marker_factor)
            if trace.marker.line is not None:
                base_mw = 1.0 if trace.marker.line.width is None else float(trace.marker.line.width)
                trace.marker.line.width = max(0.8, base_mw * stroke_factor)
        if hasattr(trace, "error_y") and trace.error_y is not None:
            base_th = 1.2 if trace.error_y.thickness is None else float(trace.error_y.thickness)
            trace.error_y.thickness = max(0.8, base_th * error_factor)
            if trace.error_y.width is not None:
                trace.error_y.width = max(1.0, float(trace.error_y.width) * error_factor)
        if hasattr(trace, "error_x") and trace.error_x is not None:
            base_th = 1.2 if trace.error_x.thickness is None else float(trace.error_x.thickness)
            trace.error_x.thickness = max(0.8, base_th * error_factor)
            if trace.error_x.width is not None:
                trace.error_x.width = max(1.0, float(trace.error_x.width) * error_factor)

    return out


def autoscale_figure_to_data(
    fig: go.Figure,
    x_values: np.ndarray,
    y_values: np.ndarray,
    sigma_y_values: np.ndarray,
    y_axis_type: str,
) -> go.Figure:
    """Return figure copy with x/y ranges recalculated from data and y-errors."""
    out = go.Figure(fig)

    x_vals = np.asarray(x_values, dtype=float)
    y_vals = np.asarray(y_values, dtype=float)
    s_vals = np.asarray(sigma_y_values, dtype=float)

    x_vals = x_vals[np.isfinite(x_vals)]
    y_vals = y_vals[np.isfinite(y_vals)]
    s_vals = s_vals[np.isfinite(s_vals)]
    if x_vals.size == 0 or y_vals.size == 0:
        return out

    x_min = float(np.min(x_vals))
    x_max = float(np.max(x_vals))
    x_span = x_max - x_min
    x_pad = max(1e-12, x_span * 0.05)
    if x_span <= 0:
        x_pad = max(1.0, abs(x_min) * 0.05)
    out.update_xaxes(autorange=False, range=[x_min - x_pad, x_max + x_pad])

    if s_vals.size != y_vals.size:
        s_vals = np.zeros_like(y_vals)

    if y_axis_type == "log":
        y_low = y_vals - s_vals
        y_high = y_vals + s_vals
        positive_low = y_low[y_low > 0]
        positive_y = y_vals[y_vals > 0]
        if positive_low.size == 0:
            if positive_y.size == 0:
                return out
            y_min = float(np.min(positive_y)) * 0.95
        else:
            y_min = float(np.min(positive_low)) * 0.95
        y_max = float(np.max(y_high)) * 1.05
        if y_max <= y_min:
            y_max = y_min * 10.0
        out.update_yaxes(autorange=False, range=[np.log10(y_min), np.log10(y_max)])
    else:
        y_low = y_vals - s_vals
        y_high = y_vals + s_vals
        y_min = float(np.min(y_low))
        y_max = float(np.max(y_high))
        y_span = y_max - y_min
        y_pad = max(1e-12, y_span * 0.06)
        if y_span <= 0:
            y_pad = max(1.0, abs(y_min) * 0.05)
        out.update_yaxes(autorange=False, range=[y_min - y_pad, y_max + y_pad])

    return out


def build_summary_text(
    raw_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    fit: dict[str, float],
    fit_model: str,
    fit_prefactor: float | None,
    fit_triangle_slope: float | None,
    error: dict[str, float] | None,
    final_slope_text: str | None,
    lang: str = "de",
) -> str:
    """Build markdown/text summary for export."""
    lines: list[str] = []
    lines.append(translate(lang, "summary.title"))
    lines.append(translate(lang, "summary.generated", timestamp=datetime.now().isoformat(timespec="seconds")))
    lines.append("")
    lines.append(translate(lang, "summary.raw_data"))
    lines.append(raw_df.to_csv(index=False).strip())
    lines.append("")
    lines.append(translate(lang, "summary.analysis_data"))
    lines.append(analysis_df.to_csv(index=False).strip())
    lines.append("")
    lines.append(translate(lang, "summary.fit"))
    if fit_model == "exp":
        lines.append("ln(y) = a*x + b_ln")
        lines.append("y = k*exp(a*x)")
        lines.append(f"a_fit = {fit['k_fit']:.{SUMMARY_FLOAT_PRECISION}g}")
        lines.append(f"b_ln_fit = {fit['l_fit']:.{SUMMARY_FLOAT_PRECISION}g}")
        if fit_prefactor is not None:
            lines.append(f"k_prefactor = {fit_prefactor:.{SUMMARY_FLOAT_PRECISION}g}")
    else:
        lines.append("y = a*x + b")
        lines.append(f"a_fit = {fit['k_fit']:.{SUMMARY_FLOAT_PRECISION}g}")
        lines.append(f"l_fit = {fit['l_fit']:.{SUMMARY_FLOAT_PRECISION}g}")
    if fit_triangle_slope is not None:
        lines.append(f"a_triangle = {fit_triangle_slope:.{SUMMARY_FLOAT_PRECISION}g}")

    if error is not None:
        lines.append("")
        lines.append(translate(lang, "summary.error_lines"))
        lines.append(f"a_min = {error['k_min']:.{SUMMARY_FLOAT_PRECISION}g}")
        lines.append(f"a_max = {error['k_max']:.{SUMMARY_FLOAT_PRECISION}g}")
        lines.append(f"delta_a = {error['delta_k']:.{SUMMARY_FLOAT_PRECISION}g}")

    if final_slope_text:
        lines.append("")
        lines.append(translate(lang, "summary.recommended_slope"))
        lines.append(final_slope_text)

    return "\n".join(lines)
