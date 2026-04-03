"""Streamlit app for linear physics lab plotting with error bars and error lines."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
import hashlib
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.calculations import (
    centroid_error_lines,
    exponential_regression,
    format_final_slope,
    linear_regression,
    logarithmic_transform_with_uncertainty,
    protocol_endpoint_error_lines,
)
from src.config import (
    APP_TITLE,
    DEFAULT_ERROR_LINE_COLOR,
    DEFAULT_FIT_COLOR,
    FAVICON_PATH,
    LOGO_DARKBLUE_PATH,
    LOGO_WHITE_PATH,
)
from src.data_io import (
    dataframe_to_csv_bytes,
    get_sample_dataframe,
    get_statistics_sample_dataframe,
    load_table_file,
    prepare_measurement_data,
)
from src.export_utils import (
    PlotTextBlockLayout,
    autoscale_figure_to_data as _autoscale_figure_to_data,
    build_summary_text as _build_summary_text,
    mm_to_px as _mm_to_px,
    paper_size_mm as _paper_size_mm,
    place_plot_text_block as _place_plot_text_block,
    scale_figure_for_export as _scale_figure_for_export,
    scaled_text_font_size_for_export as _scaled_text_font_size_for_export,
)
from src.i18n import LANGUAGE_NAMES, translate
from src.geometry import auto_triangle_points, custom_points_from_x
from src.statistics import describe_distribution, normal_curve_points
from src.plotting import (
    LineStyle,
    PlotStyle,
    add_exponential_line,
    add_line,
    add_slope_triangle,
    create_base_figure,
    figure_to_image_bytes,
    visible_x_range,
)
from src.ui_helpers import (
    auto_axis_labels as _auto_axis_labels,
    auto_triangle_delta_symbols as _auto_triangle_delta_symbols,
    auto_line_labels as _auto_line_labels,
    error_line_help_text as _error_line_help_text,
    format_exponential_equation as _format_exponential_equation,
    format_linear_equation as _format_linear_equation,
    fit_line_help_text as _fit_line_help_text,
    safe_default_index as _safe_default_index,
    to_plot_math_text as _to_plot_math_text,
)
from src.ui_state import (
    clear_session_snapshot as _clear_session_snapshot,
    init_session_state as _init_session_state,
    reset_view_state as _reset_view_state,
    restore_session_snapshot as _restore_session_snapshot,
    save_session_snapshot as _save_session_snapshot,
)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else None,
    layout="wide",
)

EXP_ERROR_MIN_COLOR = "#ff8c00"  # orange
EXP_ERROR_MAX_COLOR = "#ffd700"  # yellow
EXP_ERROR_MEAN_COLOR = "#1f77b4"  # blue


_restore_session_snapshot(st.session_state)
_init_session_state(st.session_state, get_sample_dataframe())


def _normalize_legacy_selection(key: str, mapping: dict[str, str] | None = None) -> None:
    """Convert legacy persisted widget values to current internal option keys."""
    value = st.session_state.get(key)
    if isinstance(value, tuple) and value:
        st.session_state[key] = value[0]
        return
    if isinstance(value, str) and mapping and value in mapping:
        st.session_state[key] = mapping[value]


def _t(key: str, **kwargs: object) -> str:
    """Translate UI strings from current session language."""
    return translate(str(st.session_state.get("language", "de")), key, **kwargs)


def _json_default(value: object) -> object:
    """Serialize numpy-like values for export cache signatures."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _export_signature(fig: go.Figure, **payload: object) -> str:
    """Build a stable signature for an on-demand export artifact."""
    digest = hashlib.sha256()
    digest.update(fig.to_json().encode("utf-8"))
    digest.update(json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8"))
    return digest.hexdigest()


def _export_plot_text_font_size(
    preview_figure: go.Figure,
    export_figure: go.Figure,
    preview_box_font_size: int,
    manual_layout: bool,
    fallback_font_size: int,
) -> int:
    """Keep plot info-box text visually consistent between preview and export."""
    export_base_font_size = (
        float(export_figure.layout.font.size)
        if export_figure.layout.font is not None and export_figure.layout.font.size is not None
        else float(fallback_font_size)
    )
    if not manual_layout:
        return int(round(export_base_font_size))

    preview_base_font_size = (
        float(preview_figure.layout.font.size)
        if preview_figure.layout.font is not None and preview_figure.layout.font.size is not None
        else float(fallback_font_size)
    )
    return _scaled_text_font_size_for_export(
        requested_font_size=int(preview_box_font_size),
        preview_base_font_size=preview_base_font_size,
        export_base_font_size=export_base_font_size,
    )


def _render_on_demand_image_export(
    *,
    cache_key: str,
    prepare_label: str,
    spinner_label: str,
    download_label: str,
    file_name: str,
    mime: str,
    signature: str,
    build_bytes: Callable[[], bytes],
    unavailable_message_key: str,
) -> None:
    """Prepare heavy image exports only when explicitly requested."""
    state_key = f"_export_cache_{cache_key}"
    cached = st.session_state.get(state_key)
    if not isinstance(cached, dict) or cached.get("signature") != signature:
        st.session_state.pop(state_key, None)
        cached = None

    if st.button(prepare_label, key=f"{state_key}_prepare", use_container_width=True):
        try:
            with st.spinner(spinner_label):
                export_bytes = build_bytes()
            cached = {"signature": signature, "data": export_bytes}
            st.session_state[state_key] = cached
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.session_state.pop(state_key, None)
            cached = None
            st.caption(_t(unavailable_message_key, error=exc))

    if isinstance(cached, dict) and isinstance(cached.get("data"), (bytes, bytearray)):
        st.download_button(
            download_label,
            data=bytes(cached["data"]),
            file_name=file_name,
            mime=mime,
            use_container_width=True,
            key=f"{state_key}_download",
        )


def _render_plot_text_box_controls(default_font_size: int) -> tuple[int, PlotTextBlockLayout]:
    """Render automatic/manual controls for the plot info box."""
    manual_box = st.checkbox(
        _t("plot_settings.info_box_manual"),
        value=False,
        key="plot_info_box_manual",
    )
    if not manual_box:
        return int(default_font_size), PlotTextBlockLayout()

    st.caption(_t("plot_settings.info_box_caption"))
    box_font_size = int(
        st.number_input(
            _t("plot_settings.info_box_font_size"),
            min_value=6,
            max_value=72,
            value=int(default_font_size),
            step=1,
            key="plot_info_box_font_size",
        )
    )
    box_x = float(
        st.number_input(
            _t("plot_settings.info_box_x"),
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.01,
            key="plot_info_box_x",
            format="%.2f",
        )
    )
    box_y = float(
        st.number_input(
            _t("plot_settings.info_box_y"),
            min_value=0.0,
            max_value=1.0,
            value=0.98,
            step=0.01,
            key="plot_info_box_y",
            format="%.2f",
        )
    )
    box_width = float(
        st.number_input(
            _t("plot_settings.info_box_width"),
            min_value=0.12,
            max_value=0.95,
            value=0.46,
            step=0.01,
            key="plot_info_box_width",
            format="%.2f",
        )
    )
    box_height = float(
        st.number_input(
            _t("plot_settings.info_box_height"),
            min_value=0.08,
            max_value=0.95,
            value=0.28,
            step=0.01,
            key="plot_info_box_height",
            format="%.2f",
        )
    )
    return box_font_size, PlotTextBlockLayout(
        manual=True,
        x=box_x,
        y=box_y,
        max_width=box_width,
        max_height=box_height,
    )


def _best_statistics_column(columns: list[str], dataframe: pd.DataFrame) -> str:
    """Pick the column with the most numeric values as default statistics input."""
    best_column = columns[0]
    best_count = -1
    for column in columns:
        numeric_count = int(pd.to_numeric(dataframe[column], errors="coerce").notna().sum())
        if numeric_count > best_count:
            best_column = column
            best_count = numeric_count
    return best_column


def _build_statistics_summary_text(
    raw_df: pd.DataFrame,
    stats_column: str,
    numeric_values_df: pd.DataFrame,
    stats_result: object,
    bins: int,
    normalize_density: bool,
    include_normal_fit: bool,
) -> str:
    """Build summary text for the statistics mode export."""
    lines: list[str] = []
    lines.append(_t("statistics.summary_title"))
    lines.append(_t("summary.generated", timestamp=datetime.now().isoformat(timespec="seconds")))
    lines.append("")
    lines.append(_t("summary.raw_data"))
    lines.append(raw_df.to_csv(index=False).strip())
    lines.append("")
    lines.append(_t("statistics.summary_column", column=stats_column))
    lines.append("")
    lines.append(_t("statistics.summary_numeric_values"))
    lines.append(numeric_values_df.to_csv(index=False).strip())
    lines.append("")
    lines.append(_t("statistics.summary_metrics"))
    lines.append(_t("statistics.sample_size", value=str(int(stats_result.count))))
    lines.append(_t("statistics.mean_value", value=f"{stats_result.mean:.6g}"))
    lines.append(_t("statistics.std_value", value=f"{stats_result.std:.6g}"))
    lines.append(_t("statistics.variance_value", value=f"{stats_result.variance:.6g}"))
    lines.append(_t("statistics.median_value", value=f"{stats_result.median:.6g}"))
    lines.append(_t("statistics.min_value", value=f"{stats_result.minimum:.6g}"))
    lines.append(_t("statistics.max_value", value=f"{stats_result.maximum:.6g}"))
    lines.append(_t("statistics.q1_value", value=f"{stats_result.q1:.6g}"))
    lines.append(_t("statistics.q3_value", value=f"{stats_result.q3:.6g}"))
    lines.append("")
    lines.append(_t("statistics.summary_histogram"))
    lines.append(f"bins = {int(bins)}")
    lines.append(f"density = {bool(normalize_density)}")
    lines.append("")
    lines.append(_t("statistics.summary_fit"))
    if include_normal_fit:
        lines.append(r"f(x) = 1/(\sigma\cdot sqrt(2\pi)) \cdot exp(-(x-\mu)^2 / (2\sigma^2))")
        lines.append(_t("statistics.mean_value", value=f"{stats_result.mean:.6g}"))
        lines.append(_t("statistics.std_value", value=f"{stats_result.std:.6g}"))
    else:
        lines.append(_t("statistics.summary_fit_unavailable"))
    return "\n".join(lines)


def _render_statistics_mode(edited_df: pd.DataFrame, columns: list[str]) -> None:
    """Render histogram and normal-distribution analysis for a single numeric column."""
    default_stats_column = _best_statistics_column(columns, edited_df)

    with st.sidebar:
        st.header(_t("statistics.header"))
        st.caption(_t("statistics.caption"))
        stats_column = st.selectbox(
            _t("statistics.column"),
            options=columns,
            index=_safe_default_index(columns, default_stats_column),
            key="stats_column",
            help=_t("statistics.column_help"),
        )
        bins = int(
            st.number_input(
                _t("statistics.bins"),
                min_value=4,
                max_value=100,
                value=12,
                step=1,
                key="stats_bins",
            )
        )
        normalize_density = st.checkbox(
            _t("statistics.normalize_density"),
            value=True,
            key="stats_normalize_density",
        )
        use_latex_plot = st.checkbox(
            _t("plot_settings.use_latex"),
            value=True,
            key="use_latex_plot",
            help=_t("plot_settings.use_latex_help"),
        )
        auto_axis_labels = st.checkbox(
            _t("statistics.auto_axis_labels"),
            value=True,
            key="stats_auto_axis_labels",
        )
        default_x_label = str(stats_column).strip() or "x"
        default_y_label = (
            _t("statistics.default_y_label_density")
            if normalize_density
            else _t("statistics.default_y_label_count")
        )
        if auto_axis_labels:
            st.session_state["stats_x_label_input"] = default_x_label
            st.session_state["stats_y_label_input"] = default_y_label
        x_label = st.text_input(
            _t("statistics.x_axis_label"),
            key="stats_x_label_input",
            disabled=auto_axis_labels,
            help=_t("plot_settings.x_axis_help"),
        )
        y_label = st.text_input(
            _t("statistics.y_axis_label"),
            key="stats_y_label_input",
            disabled=auto_axis_labels,
            help=_t("plot_settings.y_axis_help"),
        )
        if auto_axis_labels:
            x_label = default_x_label
            y_label = default_y_label
        st.caption(_t("plot_settings.axis_tip"))

        use_separate_font_sizes = st.checkbox(
            _t("plot_settings.separate_fonts"),
            value=False,
            key="use_separate_fonts",
        )
        if use_separate_font_sizes:
            base_font_size = st.number_input(
                _t("plot_settings.base_font_size"),
                min_value=8,
                max_value=36,
                value=14,
                step=1,
                key="base_font_size",
            )
            axis_title_font_size = st.number_input(
                _t("plot_settings.axis_title_font_size"),
                min_value=8,
                max_value=42,
                value=16,
                step=1,
                key="axis_title_font_size",
            )
            tick_font_size = st.number_input(
                _t("plot_settings.tick_font_size"),
                min_value=8,
                max_value=30,
                value=12,
                step=1,
                key="tick_font_size",
            )
            annotation_font_size = st.number_input(
                _t("plot_settings.annotation_font_size"),
                min_value=8,
                max_value=30,
                value=12,
                step=1,
                key="annotation_font_size",
            )
        else:
            global_font_size = st.number_input(
                _t("plot_settings.font_size_all"),
                min_value=8,
                max_value=42,
                value=14,
                step=1,
                key="global_font_size",
            )
            base_font_size = int(global_font_size)
            axis_title_font_size = int(global_font_size)
            tick_font_size = int(global_font_size)
            annotation_font_size = int(global_font_size)

        plot_info_box_font_size, plot_info_box_layout = _render_plot_text_box_controls(int(annotation_font_size))

        show_grid = st.checkbox(_t("plot_settings.show_grid"), value=True, key="show_grid")
        x_tick_decimals = st.number_input(
            _t("plot_settings.x_decimals"),
            min_value=0,
            max_value=8,
            value=2,
            step=1,
            key="x_tick_decimals",
        )
        y_tick_decimals = st.number_input(
            _t("plot_settings.y_decimals"),
            min_value=0,
            max_value=8,
            value=3,
            step=1,
            key="y_tick_decimals",
        )

        show_normal_fit = st.checkbox(
            _t("statistics.show_normal_fit"),
            value=True,
            key="stats_show_normal_fit",
        )
        show_formula_box = st.checkbox(
            _t("statistics.show_formula_box"),
            value=True,
            disabled=not show_normal_fit,
            key="stats_show_formula_box",
        )
        show_mean_line = st.checkbox(
            _t("statistics.show_mean_line"),
            value=True,
            key="stats_show_mean_line",
        )
        show_std_lines = st.checkbox(
            _t("statistics.show_std_lines"),
            value=True,
            key="stats_show_std_lines",
        )
        show_two_sigma = st.checkbox(
            _t("statistics.show_two_sigma"),
            value=True,
            disabled=not show_std_lines,
            key="stats_show_two_sigma",
        )
        show_three_sigma = st.checkbox(
            _t("statistics.show_three_sigma"),
            value=False,
            disabled=not show_std_lines,
            key="stats_show_three_sigma",
        )
        histogram_color = st.color_picker(
            _t("statistics.histogram_color"),
            value="#7aa6ff",
            key="stats_histogram_color",
        )
        stats_fit_color = st.color_picker(
            _t("statistics.fit_color"),
            value="#2ca02c",
            key="stats_fit_color",
        )
        stats_mean_color = st.color_picker(
            _t("statistics.mean_color"),
            value="#d62728",
            key="stats_mean_color",
        )
        stats_std_color = st.color_picker(
            _t("statistics.std_color"),
            value="#ff8c00",
            key="stats_std_color",
        )

    numeric_series = pd.to_numeric(edited_df[stats_column], errors="coerce")
    numeric_values = numeric_series.dropna().to_numpy(dtype=float)
    dropped_count = int(numeric_series.isna().sum())

    if numeric_values.size == 0:
        st.error(_t("statistics.no_numeric_values"))
        st.stop()
    if numeric_values.size < 2:
        st.error(_t("statistics.not_enough_values"))
        st.stop()
    if dropped_count > 0:
        st.info(_t("statistics.numeric_rows_used", used=str(int(numeric_values.size)), dropped=str(dropped_count)))

    stats_result = describe_distribution(numeric_values)
    constant_data = bool(np.isclose(stats_result.std, 0.0))
    if constant_data and show_normal_fit:
        st.info(_t("statistics.fit_unavailable_constant"))

    hist_range = None
    if np.isclose(stats_result.minimum, stats_result.maximum):
        pad = max(1.0, abs(stats_result.minimum) * 0.05, 0.5)
        hist_range = (stats_result.minimum - pad, stats_result.maximum + pad)

    hist_values, bin_edges = np.histogram(
        numeric_values,
        bins=int(bins),
        density=bool(normalize_density),
        range=hist_range,
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    avg_bin_width = float(np.mean(bin_widths)) if bin_widths.size else 1.0

    normal_curve_x: np.ndarray | None = None
    normal_curve_y: np.ndarray | None = None
    include_normal_fit = bool(show_normal_fit and not constant_data)
    if include_normal_fit:
        normal_curve_x, normal_curve_y = normal_curve_points(
            numeric_values,
            mean=stats_result.mean,
            std=stats_result.std,
        )
        if not normalize_density:
            normal_curve_y = normal_curve_y * float(numeric_values.size) * avg_bin_width

    x_positions = [float(bin_edges[0]), float(bin_edges[-1])]
    if normal_curve_x is not None:
        x_positions.extend([float(normal_curve_x[0]), float(normal_curve_x[-1])])
    if show_mean_line:
        x_positions.append(float(stats_result.mean))
    if show_std_lines and not constant_data:
        sigma_multiples = [1]
        if show_two_sigma:
            sigma_multiples.append(2)
        if show_three_sigma:
            sigma_multiples.append(3)
        for multiple in sigma_multiples:
            x_positions.append(float(stats_result.mean - multiple * stats_result.std))
            x_positions.append(float(stats_result.mean + multiple * stats_result.std))

    x_min_plot = float(min(x_positions))
    x_max_plot = float(max(x_positions))
    if np.isclose(x_min_plot, x_max_plot):
        x_pad = max(1.0, abs(x_min_plot) * 0.05, 0.5)
    else:
        x_pad = (x_max_plot - x_min_plot) * 0.05
    x_min_plot -= x_pad
    x_max_plot += x_pad

    y_candidates = [float(np.max(hist_values)) if hist_values.size else 0.0]
    if normal_curve_y is not None and normal_curve_y.size:
        y_candidates.append(float(np.max(normal_curve_y)))
    y_top = max(1.0, max(y_candidates) * 1.12 if max(y_candidates) > 0 else 1.0)

    rendered_x_label = _to_plot_math_text(x_label, use_latex_plot)
    rendered_y_label = _to_plot_math_text(y_label, use_latex_plot)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist_values,
            width=bin_widths,
            marker={"color": histogram_color, "line": {"color": histogram_color, "width": 1.0}},
            opacity=0.72,
            name=_to_plot_math_text(_t("statistics.histogram_label"), use_latex_plot),
            hovertemplate=(
                f"{rendered_x_label}: %{{x:.{int(x_tick_decimals)}f}}<br>"
                f"{rendered_y_label}: %{{y:.{int(y_tick_decimals)}f}}<extra></extra>"
            ),
        )
    )

    if normal_curve_x is not None and normal_curve_y is not None:
        fig.add_trace(
            go.Scatter(
                x=normal_curve_x,
                y=normal_curve_y,
                mode="lines",
                line={"color": stats_fit_color, "width": 2.6},
                name=_to_plot_math_text(_t("statistics.normal_fit_label"), use_latex_plot),
            )
        )

    def _add_vertical_marker(x_value: float, label: str, color: str, dash: str, legend_group: str) -> None:
        fig.add_trace(
            go.Scatter(
                x=[x_value, x_value],
                y=[0.0, y_top],
                mode="lines",
                line={"color": color, "width": 1.8, "dash": dash},
                name=_to_plot_math_text(label, use_latex_plot),
                legendgroup=legend_group,
                hovertemplate=f"x = {x_value:.6g}<extra></extra>",
            )
        )

    if show_mean_line:
        _add_vertical_marker(
            float(stats_result.mean),
            _t("statistics.mean_legend"),
            stats_mean_color,
            "dash",
            "stats_mean",
        )

    if show_std_lines and not constant_data:
        sigma_multiples = [1]
        if show_two_sigma:
            sigma_multiples.append(2)
        if show_three_sigma:
            sigma_multiples.append(3)
        for multiple in sigma_multiples:
            sigma_label = _t("statistics.sigma_legend", multiple=str(multiple))
            for sign, dash_style, show_legend in ((-1.0, "dot", True), (1.0, "dot", False)):
                fig.add_trace(
                    go.Scatter(
                        x=[float(stats_result.mean + sign * multiple * stats_result.std)] * 2,
                        y=[0.0, y_top],
                        mode="lines",
                        line={"color": stats_std_color, "width": 1.5, "dash": dash_style},
                        name=_to_plot_math_text(sigma_label, use_latex_plot),
                        legendgroup=f"stats_sigma_{multiple}",
                        showlegend=show_legend,
                        hovertemplate=(
                            f"x = {float(stats_result.mean + sign * multiple * stats_result.std):.6g}<extra></extra>"
                        ),
                    )
                )

    fig.update_layout(
        template="plotly_white",
        xaxis_title=rendered_x_label,
        yaxis_title=rendered_y_label,
        font={"size": int(base_font_size)},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(255,255,255,0.88)",
        },
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
        bargap=0.04,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(
        showgrid=show_grid,
        gridcolor="rgba(120,120,120,0.32)",
        tickformat=f".{int(x_tick_decimals)}f",
        title_font={"size": int(axis_title_font_size)},
        tickfont={"size": int(tick_font_size)},
        range=[x_min_plot, x_max_plot],
    )
    fig.update_yaxes(
        showgrid=show_grid,
        gridcolor="rgba(120,120,120,0.32)",
        tickformat=f".{int(y_tick_decimals)}f",
        title_font={"size": int(axis_title_font_size)},
        tickfont={"size": int(tick_font_size)},
        range=[0.0, y_top],
    )

    plot_info_lines: list[str] = []
    if show_formula_box and include_normal_fit:
        plot_info_lines = [
            _t("statistics.formula_title"),
            "f(x) = 1/(σ·sqrt(2π)) · exp(-(x-μ)² / (2σ²))",
            f"\\mu = {stats_result.mean:.6g}",
            f"\\sigma = {stats_result.std:.6g}",
        ]
    _place_plot_text_block(
        fig,
        [_to_plot_math_text(line, use_latex_plot) for line in plot_info_lines],
        font_size=int(plot_info_box_font_size),
        layout=plot_info_box_layout,
    )

    plot_col, output_col = st.columns([2.2, 1.1])
    with plot_col:
        st.subheader(_t("main.plot"))
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with output_col:
        st.subheader(_t("statistics.output_header"))
        st.latex(r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)")
        st.write(_t("statistics.sample_size", value=str(int(stats_result.count))))
        st.latex(rf"\mu = {stats_result.mean:.6g}")
        st.latex(rf"\sigma = {stats_result.std:.6g}")
        st.write(_t("statistics.variance_value", value=f"{stats_result.variance:.6g}"))
        st.write(_t("statistics.median_value", value=f"{stats_result.median:.6g}"))
        st.write(_t("statistics.min_value", value=f"{stats_result.minimum:.6g}"))
        st.write(_t("statistics.max_value", value=f"{stats_result.maximum:.6g}"))
        st.write(_t("statistics.q1_value", value=f"{stats_result.q1:.6g}"))
        st.write(_t("statistics.q3_value", value=f"{stats_result.q3:.6g}"))
        if not include_normal_fit:
            st.caption(_t("statistics.summary_fit_unavailable"))

    st.subheader(_t("export.header"))
    with st.sidebar:
        st.header(_t("export.sidebar_header"))
        export_base = st.text_input(_t("export.filename_prefix"), value="statistics_plot", key="export_base")
        png_paper = st.selectbox(_t("export.paper_size"), options=["A4", "A5", "Letter"], index=0, key="png_paper")
        png_orientation = st.selectbox(
            _t("export.orientation"),
            options=["portrait", "landscape"],
            index=1,
            key="png_orientation",
            format_func=lambda value: _t(f"orientation.{value}"),
        )
        png_dpi = st.number_input(_t("export.png_dpi"), min_value=72, max_value=600, value=300, step=1, key="png_dpi")
        png_scale = st.number_input(_t("export.png_scale"), min_value=1.0, max_value=4.0, value=1.0, step=0.1, key="png_scale")
        png_word_like = st.checkbox(
            _t("export.word_like_text"),
            value=True,
            key="png_word_like",
        )
        png_text_pt = st.number_input(
            _t("export.target_text_size"),
            min_value=6.0,
            max_value=36.0,
            value=14.0,
            step=0.5,
            disabled=not png_word_like,
            key="png_text_pt",
        )
        png_visual_scale = st.number_input(
            _t("export.extra_visual_scale"),
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.1,
            key="png_visual_scale",
        )

    numeric_values_df = pd.DataFrame({stats_column: numeric_values})
    summary_text = _build_statistics_summary_text(
        raw_df=edited_df,
        stats_column=stats_column,
        numeric_values_df=numeric_values_df,
        stats_result=stats_result,
        bins=int(bins),
        normalize_density=bool(normalize_density),
        include_normal_fit=include_normal_fit,
    )

    paper_w_mm, paper_h_mm = _paper_size_mm(png_paper)
    if png_orientation == "landscape":
        paper_w_mm, paper_h_mm = paper_h_mm, paper_w_mm
    base_export_dpi = int(png_dpi)
    png_width = _mm_to_px(paper_w_mm, base_export_dpi)
    png_height = _mm_to_px(paper_h_mm, base_export_dpi)
    png_export_scale = float(max(0.2, png_scale))

    export_cols = st.columns(5)

    with export_cols[0]:
        st.download_button(
            _t("statistics.export_values_csv"),
            data=dataframe_to_csv_bytes(numeric_values_df),
            file_name=f"{export_base}_numeric_values.csv",
            mime="text/csv",
            use_container_width=True,
        )

    def _build_statistics_export_figure() -> go.Figure:
        export_figure = go.Figure(fig)
        export_figure.update_layout(width=png_width, height=png_height)
        export_figure = _scale_figure_for_export(
            export_figure,
            visual_scale=float(png_visual_scale),
            target_text_pt=float(png_text_pt) if png_word_like else None,
            base_export_dpi=base_export_dpi,
        )
        export_box_font_size = _export_plot_text_font_size(
            preview_figure=fig,
            export_figure=export_figure,
            preview_box_font_size=int(plot_info_box_font_size),
            manual_layout=bool(plot_info_box_layout.manual),
            fallback_font_size=int(annotation_font_size),
        )
        _place_plot_text_block(
            export_figure,
            [_to_plot_math_text(line, use_latex_plot) for line in plot_info_lines],
            font_size=export_box_font_size,
            layout=plot_info_box_layout,
        )
        return export_figure

    with export_cols[1]:
        _render_on_demand_image_export(
            cache_key="statistics_png",
            prepare_label=_t("export.prepare", format="PNG"),
            spinner_label=_t("export.preparing", format="PNG"),
            download_label=_t("export.download_png", paper=png_paper, orientation=_t(f"orientation.{png_orientation}")),
            file_name=f"{export_base}.png",
            mime="image/png",
            signature=_export_signature(
                fig,
                mode="statistics",
                format="png",
                width=png_width,
                height=png_height,
                scale=png_export_scale,
                base_export_dpi=base_export_dpi,
                word_like=bool(png_word_like),
                target_text_pt=float(png_text_pt) if png_word_like else None,
                visual_scale=float(png_visual_scale),
                plot_info_lines=[_to_plot_math_text(line, use_latex_plot) for line in plot_info_lines],
                plot_info_box_layout=asdict(plot_info_box_layout),
                plot_info_box_font_size=int(plot_info_box_font_size),
                annotation_font_size=int(annotation_font_size),
                use_latex_plot=bool(use_latex_plot),
            ),
            build_bytes=lambda: figure_to_image_bytes(
                _build_statistics_export_figure(),
                "png",
                width=png_width,
                height=png_height,
                scale=png_export_scale,
            ),
            unavailable_message_key="export.png_unavailable",
        )

    with export_cols[2]:
        _render_on_demand_image_export(
            cache_key="statistics_svg",
            prepare_label=_t("export.prepare", format="SVG"),
            spinner_label=_t("export.preparing", format="SVG"),
            download_label=_t("export.download_svg"),
            file_name=f"{export_base}.svg",
            mime="image/svg+xml",
            signature=_export_signature(
                fig,
                mode="statistics",
                format="svg",
                width=png_width,
                height=png_height,
                scale=1.0,
                base_export_dpi=base_export_dpi,
                word_like=bool(png_word_like),
                target_text_pt=float(png_text_pt) if png_word_like else None,
                visual_scale=float(png_visual_scale),
                plot_info_lines=[_to_plot_math_text(line, use_latex_plot) for line in plot_info_lines],
                plot_info_box_layout=asdict(plot_info_box_layout),
                plot_info_box_font_size=int(plot_info_box_font_size),
                annotation_font_size=int(annotation_font_size),
                use_latex_plot=bool(use_latex_plot),
            ),
            build_bytes=lambda: figure_to_image_bytes(
                _build_statistics_export_figure(),
                "svg",
                width=png_width,
                height=png_height,
                scale=1.0,
            ),
            unavailable_message_key="export.svg_unavailable",
        )

    with export_cols[3]:
        _render_on_demand_image_export(
            cache_key="statistics_pdf",
            prepare_label=_t("export.prepare", format="PDF"),
            spinner_label=_t("export.preparing", format="PDF"),
            download_label=_t("export.download_pdf", paper=png_paper, orientation=_t(f"orientation.{png_orientation}")),
            file_name=f"{export_base}.pdf",
            mime="application/pdf",
            signature=_export_signature(
                fig,
                mode="statistics",
                format="pdf",
                width=png_width,
                height=png_height,
                scale=1.0,
                base_export_dpi=base_export_dpi,
                word_like=bool(png_word_like),
                target_text_pt=float(png_text_pt) if png_word_like else None,
                visual_scale=float(png_visual_scale),
                plot_info_lines=[_to_plot_math_text(line, use_latex_plot) for line in plot_info_lines],
                plot_info_box_layout=asdict(plot_info_box_layout),
                plot_info_box_font_size=int(plot_info_box_font_size),
                annotation_font_size=int(annotation_font_size),
                use_latex_plot=bool(use_latex_plot),
            ),
            build_bytes=lambda: figure_to_image_bytes(
                _build_statistics_export_figure(),
                "pdf",
                width=png_width,
                height=png_height,
                scale=1.0,
            ),
            unavailable_message_key="export.pdf_unavailable",
        )

    with export_cols[4]:
        st.download_button(
            _t("export.download_summary_md"),
            data=summary_text.encode("utf-8"),
            file_name=f"{export_base}_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.download_button(
        _t("export.download_summary_txt"),
        data=summary_text.encode("utf-8"),
        file_name=f"{export_base}_summary.txt",
        mime="text/plain",
    )

    st.session_state["_prefs"] = {
        "app_mode": "statistics",
        "language": st.session_state.get("language", "de"),
        "stats_column": stats_column,
        "stats_bins": int(bins),
        "stats_normalize_density": bool(normalize_density),
        "use_latex_plot": bool(use_latex_plot),
        "stats_auto_axis_labels": bool(auto_axis_labels),
        "stats_x_label_input": x_label,
        "stats_y_label_input": y_label,
        "use_separate_fonts": bool(use_separate_font_sizes),
        "global_font_size": int(global_font_size) if not use_separate_font_sizes else int(base_font_size),
        "base_font_size": int(base_font_size),
        "axis_title_font_size": int(axis_title_font_size),
        "tick_font_size": int(tick_font_size),
        "annotation_font_size": int(annotation_font_size),
        "show_grid": bool(show_grid),
        "x_tick_decimals": int(x_tick_decimals),
        "y_tick_decimals": int(y_tick_decimals),
        "stats_show_normal_fit": bool(show_normal_fit),
        "stats_show_formula_box": bool(show_formula_box),
        "stats_show_mean_line": bool(show_mean_line),
        "stats_show_std_lines": bool(show_std_lines),
        "stats_show_two_sigma": bool(show_two_sigma),
        "stats_show_three_sigma": bool(show_three_sigma),
        "stats_histogram_color": histogram_color,
        "stats_fit_color": stats_fit_color,
        "stats_mean_color": stats_mean_color,
        "stats_std_color": stats_std_color,
        "plot_info_box_manual": bool(plot_info_box_layout.manual),
        "plot_info_box_font_size": int(plot_info_box_font_size),
        "plot_info_box_x": float(st.session_state.get("plot_info_box_x", 0.02)),
        "plot_info_box_y": float(st.session_state.get("plot_info_box_y", 0.98)),
        "plot_info_box_width": float(st.session_state.get("plot_info_box_width", 0.46)),
        "plot_info_box_height": float(st.session_state.get("plot_info_box_height", 0.28)),
        "export_base": export_base,
        "png_paper": png_paper,
        "png_orientation": png_orientation,
        "png_dpi": int(png_dpi),
        "png_scale": float(png_scale),
        "png_word_like": bool(png_word_like),
        "png_text_pt": float(png_text_pt),
        "png_visual_scale": float(png_visual_scale),
    }
    _save_session_snapshot(st.session_state)
    st.stop()


def _apply_sidebar_brand_css() -> None:
    """Make the sidebar brand area flush with the top sidebar edges."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 0 !important;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 0 !important;
            padding-bottom: 1rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: 0 !important;
        }

        [data-testid="stSidebar"] .sidebar-brand-shell {
            margin: -1rem -1rem 0.85rem -1rem;
            padding: 0.2rem 0.85rem 0.3rem 0.85rem;
            background: linear-gradient(135deg, rgb(2, 8, 24) 0%, rgb(8, 24, 44) 100%);
            border-bottom: 1px solid rgba(94, 223, 255, 0.12);
            overflow: hidden;
            position: relative;
            top: -0.35rem;
        }

        [data-testid="stSidebar"] .sidebar-brand-shell img {
            width: 100%;
            max-width: 100%;
            height: auto;
            display: block;
        }

        [data-testid="stSidebar"] .sidebar-brand-credit {
            margin: 0.15rem 0 0.65rem 0;
            color: rgba(71, 85, 105, 0.78);
            font-size: 0.78rem;
            letter-spacing: 0.01em;
            padding: 0 0.85rem;
        }

        [data-testid="stDeployButton"],
        .stAppDeployButton,
        button[kind="header"]:has([data-testid="stDeployButton"]) {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_brand_header() -> None:
    """Render the provided logo inside a matched dark header panel."""
    logo_path = LOGO_DARKBLUE_PATH if LOGO_DARKBLUE_PATH.exists() else LOGO_WHITE_PATH
    if not logo_path.exists():
        return

    encoded_logo = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <div class="sidebar-brand-shell">
            <img
                src="data:image/png;base64,{encoded_logo}"
                alt="{APP_TITLE} logo"
            />
        </div>
        <div class="sidebar-brand-credit">made by Mihai Cazac</div>
        """,
        unsafe_allow_html=True,
    )


_normalize_legacy_selection("mapping_mode", {"Simple": "simple", "Advanced": "advanced"})
_normalize_legacy_selection("y_scale_mode")
_normalize_legacy_selection("log_decade_mode")
_normalize_legacy_selection("fit_model")
_normalize_legacy_selection("error_line_mode_widget")
_normalize_legacy_selection("error_line_mode")
_normalize_legacy_selection("png_orientation")
_normalize_legacy_selection("app_mode", {"physics": "normal"})
_apply_sidebar_brand_css()

with st.sidebar:
    _render_brand_header()
    st.header(_t("settings.header"))
    st.selectbox(
        _t("settings.language"),
        options=["de", "en"],
        format_func=lambda code: LANGUAGE_NAMES.get(code, code),
        key="language",
    )
    st.selectbox(
        _t("settings.mode"),
        options=["normal", "statistics"],
        format_func=lambda mode: _t(f"app_mode.{mode}"),
        key="app_mode",
    )

    st.header(_t("data_source.header"))
    st.checkbox(
        _t("data_source.remember_settings"),
        key="remember_settings",
        help=_t("data_source.remember_settings_help"),
    )
    if st.button(_t("data_source.forget_saved_settings"), use_container_width=True):
        _clear_session_snapshot()
        st.success(_t("data_source.saved_settings_removed"))

    sample_button_label = _t("data_source.use_statistics_sample_data") if st.session_state.get("app_mode", "normal") == "statistics" else _t("data_source.use_normal_sample_data")
    if st.button(sample_button_label, use_container_width=True):
        st.session_state["table_df"] = get_statistics_sample_dataframe() if st.session_state.get("app_mode", "normal") == "statistics" else get_sample_dataframe()
        st.session_state["uploaded_signature"] = ""
        if "table_editor" in st.session_state:
            del st.session_state["table_editor"]
        st.rerun()

    if st.button(_t("data_source.reset_view"), use_container_width=True):
        _reset_view_state(st.session_state)
        st.rerun()

    uploaded = st.file_uploader(_t("data_source.upload"), type=["csv", "xlsx", "xls", "ods"])
    if uploaded is not None:
        signature = f"{uploaded.name}-{uploaded.size}"
        if signature != st.session_state.get("uploaded_signature", ""):
            try:
                st.session_state["table_df"] = load_table_file(uploaded, uploaded.name)
                st.session_state["uploaded_signature"] = signature
                if "table_editor" in st.session_state:
                    del st.session_state["table_editor"]
                st.success(_t("data_source.loaded_file", filename=uploaded.name))
                st.rerun()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                st.error(_t("data_source.load_failed", error=exc))

st.subheader(_t("table.header"))
st.write(_t("table.description"))

edited_df = st.data_editor(
    st.session_state["table_df"],
    num_rows="dynamic",
    use_container_width=True,
    key="table_editor",
)
st.session_state["table_df"] = edited_df

columns = [str(col) for col in edited_df.columns]
if not columns:
    st.error(_t("table.add_column_error"))
    st.stop()

if st.session_state.get("app_mode", "normal") == "statistics":
    _render_statistics_mode(edited_df, columns)

none_option = "<None>"
column_options = [none_option] + columns

with st.sidebar:
    st.header(_t("mapping.header"))
    mapping_mode = st.radio(
        _t("mapping.mode"),
        options=["simple", "advanced"],
        index=0,
        key="mapping_mode",
        format_func=lambda value: _t(f"mapping_mode.{value}"),
    )
    use_zero_error = st.checkbox(
        _t("mapping.no_error_column"),
        value=False,
        key="use_zero_error",
    )

    if mapping_mode == "simple":
        st.caption(_t("mapping.simple_caption"))
        x_col = st.selectbox(
            _t("mapping.x_column"),
            options=columns,
            index=_safe_default_index(columns, "m"),
            key="x_col_simple",
        )
        y_simple_col = st.selectbox(
            _t("mapping.y_column"),
            options=columns,
            index=_safe_default_index(columns, "y", fallback=_safe_default_index(columns, "T_mean")),
            key="y_col_simple",
        )
        if use_zero_error:
            sigma_simple_col = none_option
            st.caption(_t("mapping.vertical_unc_disabled"))
        else:
            sigma_simple_col = st.selectbox(
                _t("mapping.error_column"),
                options=columns,
                index=_safe_default_index(columns, "sigma_y", fallback=_safe_default_index(columns, "sigma_T")),
                key="sigma_col_simple",
            )
        derive_y = False
        derive_sigma_y = False
        y_col_selected = y_simple_col
        sigma_y_col_selected = sigma_simple_col
        t_col_selected = none_option
        sigma_t_col_selected = none_option
    else:
        x_col = st.selectbox(
            _t("mapping.x_column"),
            options=columns,
            index=_safe_default_index(columns, "m"),
            key="x_col_adv",
        )
        derive_y = False
        y_col_selected = st.selectbox(
            _t("mapping.y_column"),
            options=columns,
            index=_safe_default_index(columns, "y", fallback=_safe_default_index(columns, "T_mean")),
            key="y_col_adv",
        )
        t_col_selected = none_option
        derive_sigma_y = False

        if use_zero_error:
            sigma_y_col_selected = none_option
            sigma_t_col_selected = none_option
            st.caption(_t("mapping.vertical_unc_disabled"))
        else:
            sigma_y_col_selected = st.selectbox(
                _t("mapping.sigma_y_column"),
                options=columns,
                index=_safe_default_index(columns, "sigma_y", fallback=_safe_default_index(columns, "sigma_T")),
                key="sigma_y_col_adv",
            )
            sigma_t_col_selected = none_option

prepared_source_df = edited_df.copy()
if use_zero_error:
    prepared_source_df["__sigma_zero__"] = 0.0
    derive_sigma_y = False
    sigma_y_col_selected = "__sigma_zero__"
    sigma_t_col_selected = none_option

try:
    prepared = prepare_measurement_data(
        source_df=prepared_source_df,
        x_col=x_col,
        derive_y=False,
        derive_sigma_y=False,
        y_col=y_col_selected,
        sigma_y_col=sigma_y_col_selected,
        t_mean_col=None,
        sigma_t_col=None,
    )
except ValueError as exc:
    st.error(_t("validation.data_error", error=exc))
    st.stop()

analysis_df = prepared.dataframe

x_data = analysis_df["x"].to_numpy(dtype=float)
y_data = analysis_df["y"].to_numpy(dtype=float)
data_x_bounds = (float(np.min(x_data)), float(np.max(x_data)))

with st.sidebar:
    st.header(_t("plot_settings.header"))
    use_latex_plot = st.checkbox(
        _t("plot_settings.use_latex"),
        value=True,
        key="use_latex_plot",
        help=_t("plot_settings.use_latex_help"),
    )
    auto_x_label, auto_y_label = _auto_axis_labels(
        x_column=x_col,
        y_column=y_col_selected,
        derive_y=False,
        t_mean_column=None,
    )
    auto_axis_labels = st.checkbox(_t("plot_settings.auto_axis_labels"), value=True, key="auto_axis_labels")
    if auto_axis_labels:
        st.session_state["x_label_input"] = auto_x_label
        st.session_state["y_label_input"] = auto_y_label
    x_label = st.text_input(
        _t("plot_settings.x_axis_label"),
        key="x_label_input",
        disabled=auto_axis_labels,
        help=_t("plot_settings.x_axis_help"),
    )
    y_label = st.text_input(
        _t("plot_settings.y_axis_label"),
        key="y_label_input",
        disabled=auto_axis_labels,
        help=_t("plot_settings.y_axis_help"),
    )
    if auto_axis_labels:
        x_label = auto_x_label
        y_label = auto_y_label
    st.caption(_t("plot_settings.axis_tip"))
    y_scale_mode = st.selectbox(
        _t("plot_settings.y_scale"),
        options=["linear", "log"],
        format_func=lambda item: _t(f"y_scale.{item}"),
        index=0,
        key="y_scale_mode",
    )
    y_axis_type = y_scale_mode
    y_log_decades: int | None = None
    if y_axis_type == "log":
        decade_mode = st.selectbox(
            _t("plot_settings.log_decades"),
            options=["auto", "1", "2"],
            format_func=lambda item: _t(f"log_decade.{item}"),
            index=0,
            key="log_decade_mode",
        )
        if decade_mode in {"1", "2"}:
            y_log_decades = int(decade_mode)
        st.caption(_t("plot_settings.log_caption"))
    use_separate_font_sizes = st.checkbox(_t("plot_settings.separate_fonts"), value=False, key="use_separate_fonts")
    if use_separate_font_sizes:
        base_font_size = st.number_input(
            _t("plot_settings.base_font_size"),
            min_value=8,
            max_value=36,
            value=14,
            step=1,
            key="base_font_size",
        )
        axis_title_font_size = st.number_input(
            _t("plot_settings.axis_title_font_size"),
            min_value=8,
            max_value=42,
            value=16,
            step=1,
            key="axis_title_font_size",
        )
        tick_font_size = st.number_input(
            _t("plot_settings.tick_font_size"),
            min_value=8,
            max_value=30,
            value=12,
            step=1,
            key="tick_font_size",
        )
        annotation_font_size = st.number_input(
            _t("plot_settings.annotation_font_size"),
            min_value=8,
            max_value=30,
            value=12,
            step=1,
            key="annotation_font_size",
        )
    else:
        global_font_size = st.number_input(
            _t("plot_settings.font_size_all"),
            min_value=8,
            max_value=42,
            value=14,
            step=1,
            key="global_font_size",
        )
        base_font_size = int(global_font_size)
        axis_title_font_size = int(global_font_size)
        tick_font_size = int(global_font_size)
        annotation_font_size = int(global_font_size)
    plot_info_box_font_size, plot_info_box_layout = _render_plot_text_box_controls(int(annotation_font_size))
    show_grid = st.checkbox(_t("plot_settings.show_grid"), value=True, key="show_grid")
    grid_mode = st.selectbox(
        _t("plot_settings.grid_mode"),
        options=["auto", "manual", "millimetric"],
        index=0,
        disabled=not show_grid,
        key="grid_mode",
        format_func=lambda value: _t(f"grid_mode.{value}"),
    )
    x_major_divisions = 10
    y_major_divisions = 10
    minor_per_major = 10
    if show_grid and grid_mode in {"manual", "millimetric"}:
        x_major_divisions = int(
            st.number_input(
                _t("plot_settings.x_major_divisions"),
                min_value=1,
                max_value=200,
                value=10,
                step=1,
                key="x_major_divisions",
            )
        )
        if y_axis_type == "linear":
            y_major_divisions = int(
                st.number_input(
                    _t("plot_settings.y_major_divisions"),
                    min_value=1,
                    max_value=200,
                    value=10,
                    step=1,
                    key="y_major_divisions",
                )
            )
    if show_grid and grid_mode == "millimetric":
        if y_axis_type == "linear":
            minor_per_major = int(
                st.number_input(
                    _t("plot_settings.minor_per_major"),
                    min_value=2,
                    max_value=50,
                    value=10,
                    step=1,
                    key="minor_per_major",
                )
            )
        else:
            st.caption(_t("plot_settings.log_minor_caption"))
    marker_size = st.number_input(_t("plot_settings.marker_size"), min_value=1.0, max_value=30.0, value=7.0, step=0.5, key="marker_size")
    error_bar_thickness = st.number_input(
        _t("plot_settings.error_bar_thickness"), min_value=0.5, max_value=10.0, value=1.8, step=0.1, key="error_bar_thickness"
    )
    error_bar_cap_width = st.number_input(
        _t("plot_settings.error_bar_cap_width"), min_value=0.0, max_value=20.0, value=6.0, step=0.5, key="error_bar_cap_width"
    )
    connect_points = st.checkbox(_t("plot_settings.connect_points"), value=False, key="connect_points")
    x_tick_decimals = st.number_input(_t("plot_settings.x_decimals"), min_value=0, max_value=8, value=1, step=1, key="x_tick_decimals")
    y_tick_decimals = st.number_input(_t("plot_settings.y_decimals"), min_value=0, max_value=8, value=2, step=1, key="y_tick_decimals")

    custom_x_range = st.checkbox(_t("plot_settings.custom_x_range"), key="custom_x_range")
    x_range = None
    if custom_x_range:
        default_x_min = float(np.min(x_data))
        default_x_max = float(np.max(x_data))
        x_min_user = st.number_input(_t("plot_settings.x_min"), value=default_x_min, format="%.6f", key="x_min_user")
        x_max_user = st.number_input(_t("plot_settings.x_max"), value=default_x_max, format="%.6f", key="x_max_user")
        x_range = (float(x_min_user), float(x_max_user))

    custom_y_range = st.checkbox(_t("plot_settings.custom_y_range"), key="custom_y_range")
    y_range = None
    if custom_y_range:
        if y_axis_type == "log":
            positive_y = y_data[y_data > 0]
            if positive_y.size == 0:
                st.error(_t("plot_settings.log_positive_required"))
                st.stop()
            default_y_min = float(np.min(positive_y))
        else:
            default_y_min = float(np.min(y_data))
        default_y_max = float(np.max(y_data))
        y_min_user = st.number_input(_t("plot_settings.y_min"), value=default_y_min, format="%.6f", key="y_min_user")
        y_max_user = st.number_input(_t("plot_settings.y_max"), value=default_y_max, format="%.6f", key="y_max_user")
        y_range = (float(y_min_user), float(y_max_user))

    st.header(_t("lines.header"))
    fit_model_ui = st.selectbox(
        _t("lines.fit_equation"),
        options=["linear", "exp"],
        format_func=lambda item: _t(f"fit_model.{item}"),
        index=1 if y_axis_type == "log" else 0,
        key="fit_model",
        help=_t("fit_model.help"),
    )
    fit_model = fit_model_ui
    show_fit_line = st.checkbox(
        _t("lines.show_fit_line"),
        value=True,
        key="show_fit_line",
        help=_fit_line_help_text(fit_model, lang=str(st.session_state.get("language", "de"))),
    )
    show_error_lines = st.checkbox(
        _t("lines.show_error_lines"),
        value=True,
        key="show_error_lines",
        help=_error_line_help_text(
            fit_model,
            st.session_state.get("error_line_mode", "protocol"),
            lang=str(st.session_state.get("language", "de")),
        ),
    )
    error_line_mode_ui = st.selectbox(
        _t("lines.error_method"),
        options=["protocol", "centroid"],
        format_func=lambda item: _t(f"error_method.{item}"),
        index=0,
        key="error_line_mode_widget",
        help=_t("error_method.help"),
    )
    st.session_state["error_line_mode"] = error_line_mode_ui
    auto_line_labels_enabled = st.checkbox(_t("lines.auto_line_labels"), value=True, key="auto_line_labels")
    fit_label_auto, error_label_max_auto, error_label_min_auto = _auto_line_labels(
        error_line_mode_ui,
        lang=str(st.session_state.get("language", "de")),
    )
    if auto_line_labels_enabled:
        st.session_state["fit_label_input"] = fit_label_auto
        st.session_state["error_label_max_input"] = error_label_max_auto
        st.session_state["error_label_min_input"] = error_label_min_auto
    fit_label = st.text_input(
        _t("lines.fit_label"),
        key="fit_label_input",
        disabled=auto_line_labels_enabled,
        help=_t("lines.label_help"),
    )
    if auto_line_labels_enabled:
        fit_label = fit_label_auto
    fit_color = st.color_picker(_t("lines.fit_color"), value=DEFAULT_FIT_COLOR, key="fit_color")
    show_fit_slope_label = st.checkbox(_t("lines.show_fit_slope_label"), value=True, key="show_fit_slope_label")
    show_line_equations_on_plot = st.checkbox(_t("lines.show_line_equations"), value=False, key="show_line_equations_on_plot")
    show_r2_on_plot = st.checkbox(_t("lines.show_r2"), value=False, key="show_r2_on_plot")
    visible_error_lines_linear: list[str] = []
    visible_error_lines_exp: list[str] = []
    if fit_model == "exp":
        error_display_mode = "upper"
        error_color = DEFAULT_ERROR_LINE_COLOR
        st.caption(_t("lines.exp_colors_caption"))
        exp_options = ["min", "max", "mean"]
        exp_default_raw = st.session_state.get("visible_error_lines_exp", exp_options)
        exp_default = [item for item in exp_default_raw if item in exp_options] or exp_options
        visible_error_lines_exp = st.multiselect(
            _t("lines.visible_exp_error_lines"),
            options=exp_options,
            default=exp_default,
            key="visible_error_lines_exp",
            format_func=lambda item: _t(f"exp_error.{item}"),
        )
    else:
        linear_options = ["k_max", "k_min"]
        legacy_error_mode = st.session_state.get("error_display_mode", "upper")
        linear_fallback = ["k_max"] if legacy_error_mode == "upper" else ["k_min"]
        linear_default_raw = st.session_state.get("visible_error_lines_linear", linear_fallback)
        linear_default = [item for item in linear_default_raw if item in linear_options] or linear_fallback
        visible_error_lines_linear = st.multiselect(
            _t("lines.visible_linear_error_lines"),
            options=linear_options,
            default=linear_default,
            key="visible_error_lines_linear",
            format_func=lambda item: _t(f"linear_error.{item}"),
        )
        if not visible_error_lines_linear:
            error_display_mode = "none"
        elif set(visible_error_lines_linear) == {"k_max"}:
            error_display_mode = "upper"
        elif set(visible_error_lines_linear) == {"k_min"}:
            error_display_mode = "lower"
        else:
            error_display_mode = "both"
        error_color = st.color_picker(_t("lines.error_color"), value=DEFAULT_ERROR_LINE_COLOR, key="error_color")
    error_label_max = st.text_input(
        _t("lines.error_label_max"),
        key="error_label_max_input",
        disabled=auto_line_labels_enabled,
        help=_t("lines.label_help"),
    )
    error_label_min = st.text_input(
        _t("lines.error_label_min"),
        key="error_label_min_input",
        disabled=auto_line_labels_enabled,
        help=_t("lines.label_help"),
    )
    if auto_line_labels_enabled:
        error_label_max = error_label_max_auto
        error_label_min = error_label_min_auto
    show_error_slope_label = st.checkbox(_t("lines.show_error_slope_label"), value=True, key="show_error_slope_label")

    st.header(_t("triangles.header"))
    show_fit_triangle = st.checkbox(_t("triangles.show_fit"), value=True, key="show_fit_triangle")
    auto_fit_points = st.checkbox(_t("triangles.auto_points"), value=True, key="auto_fit_points")
    horizontal_delta_symbol, vertical_delta_symbol = _auto_triangle_delta_symbols(x_label, y_label)
    st.caption(
        _t("triangles.auto_labels", dx=horizontal_delta_symbol, dy=vertical_delta_symbol)
    )
    show_error_triangles = st.checkbox(
        _t("triangles.show_error"),
        value=True,
        key="show_error_triangles",
    )
    triangle_x_decimals = st.number_input(
        _t("triangles.dx_decimals"),
        min_value=0,
        max_value=8,
        value=1,
        step=1,
        key="triangle_x_decimals",
    )
    triangle_y_decimals = st.number_input(
        _t("triangles.dy_decimals"),
        min_value=0,
        max_value=8,
        value=2,
        step=1,
        key="triangle_y_decimals",
    )

x_view_min, x_view_max = (
    (float(x_range[0]), float(x_range[1])) if x_range is not None else (float(np.min(x_data)), float(np.max(x_data)))
)
y_view_min, y_view_max = (
    (float(y_range[0]), float(y_range[1])) if y_range is not None else (float(np.min(y_data)), float(np.max(y_data)))
)

if y_axis_type == "log":
    if np.any(y_data <= 0):
        st.error(_t("runtime.semilog_positive"))
        st.stop()
    if y_range is not None and (y_range[0] <= 0 or y_range[1] <= 0):
        st.error(_t("runtime.custom_log_positive"))
        st.stop()
    if np.any((analysis_df["y"] - analysis_df["sigma_y"]) <= 0):
        st.info(_t("runtime.lower_error_clipped"))

sigma_y_data = analysis_df["sigma_y"].to_numpy(dtype=float)
fit_prefactor: float | None = None
try:
    if fit_model == "exp":
        fit_result = exponential_regression(x_data, y_data)
        fit_prefactor = float(np.exp(fit_result.l_fit))
        y_for_error, sigma_for_error = logarithmic_transform_with_uncertainty(y_data, sigma_y_data)
    else:
        fit_result = linear_regression(x_data, y_data)
        y_for_error, sigma_for_error = y_data, sigma_y_data
except ValueError as exc:
    st.error(_t("runtime.fit_error", error=exc))
    st.stop()

error_line_result = None
error_line_error = None
error_line_method = "centroid"
try:
    if st.session_state.get("error_line_mode", "protocol") == "protocol":
        error_line_result = protocol_endpoint_error_lines(x_data, y_for_error, sigma_for_error)
        error_line_method = "protocol"
    else:
        error_line_result = centroid_error_lines(x_data, y_for_error, sigma_for_error)
        error_line_method = "centroid"
except ValueError as exc:
    error_line_error = str(exc)

x_tick0 = None
y_tick0 = None
x_major_dtick = None
y_major_dtick = None
show_minor_grid = False
x_minor_dtick = None
y_minor_dtick = None

if show_grid and grid_mode in {"manual", "millimetric"}:
    if x_view_max > x_view_min:
        x_tick0 = x_view_min
        x_major_dtick = (x_view_max - x_view_min) / float(max(1, x_major_divisions))
        if grid_mode == "millimetric":
            show_minor_grid = True
            x_minor_dtick = x_major_dtick / float(max(1, minor_per_major))
    if y_axis_type == "linear" and y_view_max > y_view_min:
        y_tick0 = y_view_min
        y_major_dtick = (y_view_max - y_view_min) / float(max(1, y_major_divisions))
        if grid_mode == "millimetric":
            show_minor_grid = True
            y_minor_dtick = y_major_dtick / float(max(1, minor_per_major))
    elif y_axis_type == "log":
        if grid_mode == "millimetric":
            y_major_dtick = "D1"
            show_minor_grid = True
            y_minor_dtick = "D1"
        else:
            y_major_dtick = "D2"

plot_style = PlotStyle(
    x_label=_to_plot_math_text(x_label, use_latex_plot),
    y_label=_to_plot_math_text(y_label, use_latex_plot),
    show_grid=show_grid,
    measured_points_label=_t("plot.measured_points"),
    y_axis_type=y_axis_type,
    y_log_decades=y_log_decades,
    connect_points=connect_points,
    marker_size=float(marker_size),
    error_bar_thickness=float(error_bar_thickness),
    error_bar_cap_width=float(error_bar_cap_width),
    x_tick_decimals=int(x_tick_decimals),
    y_tick_decimals=int(y_tick_decimals),
    base_font_size=int(base_font_size),
    axis_title_font_size=int(axis_title_font_size),
    tick_font_size=int(tick_font_size),
    x_tick0=x_tick0,
    y_tick0=y_tick0,
    x_major_dtick=x_major_dtick,
    y_major_dtick=y_major_dtick,
    show_minor_grid=show_minor_grid,
    x_minor_dtick=x_minor_dtick,
    y_minor_dtick=y_minor_dtick,
    x_range=x_range,
    y_range=y_range,
)

fig = create_base_figure(analysis_df, plot_style)

try:
    visible_bounds = visible_x_range(x_data, custom=x_range)
except ValueError as exc:
    st.error(_t("runtime.axis_range_error", error=exc))
    st.stop()
fit_triangle_slope: float | None = None
plot_info_lines: list[str] = []
fit_label_rendered = _to_plot_math_text(fit_label, use_latex_plot)
error_label_max_rendered = _to_plot_math_text(error_label_max, use_latex_plot)
error_label_min_rendered = _to_plot_math_text(error_label_min, use_latex_plot)

if show_fit_line:
    fit_line_label = fit_label_rendered
    if show_fit_slope_label:
        fit_line_label = _to_plot_math_text(f"{fit_label} (a={fit_result.k_fit:.4g})", use_latex_plot)
    if fit_model == "exp":
        add_exponential_line(
            fig,
            slope_log=fit_result.k_fit,
            intercept_log=fit_result.l_fit,
            x_bounds=visible_bounds,
            line_style=LineStyle(color=fit_color, dash="solid", width=2.5, label=fit_line_label),
        )
        if show_line_equations_on_plot:
            plot_info_lines.append(
                _format_exponential_equation(
                    fit_label,
                    slope_log=fit_result.k_fit,
                    intercept_log=fit_result.l_fit,
                )
            )
    else:
        add_line(
            fig,
            slope=fit_result.k_fit,
            intercept=fit_result.l_fit,
            x_bounds=visible_bounds,
            line_style=LineStyle(color=fit_color, dash="solid", width=2.5, label=fit_line_label),
        )
        if show_line_equations_on_plot:
            plot_info_lines.append(
                _format_linear_equation(
                    fit_label,
                    slope=fit_result.k_fit,
                    intercept=fit_result.l_fit,
                )
            )

    if show_fit_triangle:
        if fit_model == "exp":
            st.info(_t("runtime.fit_triangle_disabled_exp"))
            show_fit_triangle = False
        if not show_fit_triangle:
            pass
        elif auto_fit_points:
            point_a, point_b = auto_triangle_points(
                slope=fit_result.k_fit,
                intercept=fit_result.l_fit,
                x_min=data_x_bounds[0],
                x_max=data_x_bounds[1],
                margin_fraction=0.08,
            )
        else:
            with st.sidebar:
                st.write(_t("runtime.custom_ab_caption"))
                x_a_fit = st.number_input(
                    "A_x",
                    value=float(data_x_bounds[0] + 0.15 * (data_x_bounds[1] - data_x_bounds[0])),
                    format="%.6f",
                )
                x_b_fit = st.number_input(
                    "B_x",
                    value=float(data_x_bounds[0] + 0.85 * (data_x_bounds[1] - data_x_bounds[0])),
                    format="%.6f",
                )
            try:
                point_a, point_b = custom_points_from_x(
                    fit_result.k_fit,
                    fit_result.l_fit,
                    x_a=x_a_fit,
                    x_b=x_b_fit,
                )
            except ValueError as exc:
                st.warning(_t("runtime.custom_fit_points_invalid", error=exc))
                point_a, point_b = auto_triangle_points(
                    slope=fit_result.k_fit,
                    intercept=fit_result.l_fit,
                    x_min=data_x_bounds[0],
                    x_max=data_x_bounds[1],
                    margin_fraction=0.08,
                )

        if show_fit_triangle:
            fit_triangle_slope = add_slope_triangle(
                fig,
                a=point_a,
                b=point_b,
                color=fit_color,
                label_prefix=fit_label_rendered,
                horizontal_symbol=horizontal_delta_symbol,
                vertical_symbol=vertical_delta_symbol,
                use_latex=use_latex_plot,
                x_decimals=int(triangle_x_decimals),
                y_decimals=int(triangle_y_decimals),
                font_size=int(annotation_font_size),
                annotate=True,
            )

error_triangle_slopes: dict[str, float] = {}

if show_error_lines and error_line_result is not None:
    if fit_model == "exp":
        visible_exp = set(visible_error_lines_exp)
        a_min = float(error_line_result.k_min)
        a_max = float(error_line_result.k_max)
        b_min = float(error_line_result.l_min)
        b_max = float(error_line_result.l_max)
        a_mean = float((a_min + a_max) / 2.0)
        b_mean = float((b_min + b_max) / 2.0)

        label_min = _t("line.exp.min")
        label_max = _t("line.exp.max")
        label_mean = _t("line.exp.mean")
        if show_error_slope_label:
            label_min = f"{label_min} (a={a_min:.4g})"
            label_max = f"{label_max} (a={a_max:.4g})"
            label_mean = f"{label_mean} (a={a_mean:.4g})"

        if "min" in visible_exp:
            add_exponential_line(
                fig,
                slope_log=a_min,
                intercept_log=b_min,
                x_bounds=visible_bounds,
                line_style=LineStyle(color=EXP_ERROR_MIN_COLOR, dash="dot", width=2.0, label=label_min),
            )
        if "max" in visible_exp:
            add_exponential_line(
                fig,
                slope_log=a_max,
                intercept_log=b_max,
                x_bounds=visible_bounds,
                line_style=LineStyle(color=EXP_ERROR_MAX_COLOR, dash="dash", width=2.0, label=label_max),
            )
        if "mean" in visible_exp:
            add_exponential_line(
                fig,
                slope_log=a_mean,
                intercept_log=b_mean,
                x_bounds=visible_bounds,
                line_style=LineStyle(color=EXP_ERROR_MEAN_COLOR, dash="solid", width=2.4, label=label_mean),
            )

        if show_line_equations_on_plot:
            if "min" in visible_exp:
                plot_info_lines.append(
                    _format_exponential_equation(
                        _t("line.eq.exp.min"),
                        slope_log=a_min,
                        intercept_log=b_min,
                    )
                )
            if "max" in visible_exp:
                plot_info_lines.append(
                    _format_exponential_equation(
                        _t("line.eq.exp.max"),
                        slope_log=a_max,
                        intercept_log=b_max,
                    )
                )
            if "mean" in visible_exp:
                plot_info_lines.append(
                    _format_exponential_equation(
                        _t("line.eq.exp.mean"),
                        slope_log=a_mean,
                        intercept_log=b_mean,
                    )
                )

        if show_error_triangles and visible_exp:
            st.info(_t("runtime.error_triangles_disabled_exp"))
    else:
        visible_linear = set(visible_error_lines_linear)
        show_k_max = "k_max" in visible_linear
        show_k_min = "k_min" in visible_linear

        if show_k_max:
            error_line_label = error_label_max
            if show_error_slope_label:
                error_line_label = _to_plot_math_text(
                    f"{error_label_max} (a={error_line_result.k_max:.4g})",
                    use_latex_plot,
                )
            else:
                error_line_label = error_label_max_rendered
            add_line(
                fig,
                slope=error_line_result.k_max,
                intercept=error_line_result.l_max,
                x_bounds=visible_bounds,
                line_style=LineStyle(
                    color=error_color,
                    dash="dash",
                    width=2.0,
                    label=error_line_label,
                ),
            )
            if show_line_equations_on_plot:
                plot_info_lines.append(
                    _format_linear_equation(
                        error_label_max,
                        slope=error_line_result.k_max,
                        intercept=error_line_result.l_max,
                    )
                )

        if show_k_min:
            error_line_label = error_label_min
            if show_error_slope_label:
                error_line_label = _to_plot_math_text(
                    f"{error_label_min} (a={error_line_result.k_min:.4g})",
                    use_latex_plot,
                )
            else:
                error_line_label = error_label_min_rendered
            add_line(
                fig,
                slope=error_line_result.k_min,
                intercept=error_line_result.l_min,
                x_bounds=visible_bounds,
                line_style=LineStyle(
                    color=error_color,
                    dash="dot",
                    width=2.0,
                    label=error_line_label,
                ),
            )
            if show_line_equations_on_plot:
                plot_info_lines.append(
                    _format_linear_equation(
                        error_label_min,
                        slope=error_line_result.k_min,
                        intercept=error_line_result.l_min,
                    )
                )
        if show_error_triangles:
            if show_k_max:
                x0, x1 = data_x_bounds
                dx = x1 - x0
                p1_max, p2_max = custom_points_from_x(
                    error_line_result.k_max,
                    error_line_result.l_max,
                    x_a=float(x0 + 0.05 * dx),
                    x_b=float(x0 + 0.95 * dx),
                )
                a_max_pt, b_max_pt = (p1_max, p2_max) if p1_max.y >= p2_max.y else (p2_max, p1_max)
                error_triangle_slopes["k_max"] = add_slope_triangle(
                    fig,
                    a=a_max_pt,
                    b=b_max_pt,
                    color=error_color,
                    label_prefix="k_max",
                    horizontal_symbol=horizontal_delta_symbol,
                    vertical_symbol=vertical_delta_symbol,
                    use_latex=use_latex_plot,
                    x_decimals=int(triangle_x_decimals),
                    y_decimals=int(triangle_y_decimals),
                    font_size=int(annotation_font_size),
                    annotate=True,
                )

            if show_k_min:
                x0, x1 = data_x_bounds
                dx = x1 - x0
                p1_min, p2_min = custom_points_from_x(
                    error_line_result.k_min,
                    error_line_result.l_min,
                    x_a=float(x0 + 0.05 * dx),
                    x_b=float(x0 + 0.95 * dx),
                )
                a_min_pt, b_min_pt = (p1_min, p2_min) if p1_min.y >= p2_min.y else (p2_min, p1_min)
                error_triangle_slopes["k_min"] = add_slope_triangle(
                    fig,
                    a=a_min_pt,
                    b=b_min_pt,
                    color=error_color,
                    label_prefix="k_min",
                    horizontal_symbol=horizontal_delta_symbol,
                    vertical_symbol=vertical_delta_symbol,
                    use_latex=use_latex_plot,
                    x_decimals=int(triangle_x_decimals),
                    y_decimals=int(triangle_y_decimals),
                    font_size=int(annotation_font_size),
                    annotate=True,
                )

if error_line_error and show_error_lines:
    st.warning(_t("runtime.error_lines_unavailable", error=error_line_error))

if show_r2_on_plot and show_fit_line:
    r_squared = float(fit_result.r_value**2)
    if fit_model == "exp":
        plot_info_lines.append(f"R²(ln y) = {r_squared:.6g}")
    else:
        plot_info_lines.append(f"R² = {r_squared:.6g}")

plot_info_lines = [_to_plot_math_text(line, use_latex_plot) for line in plot_info_lines]
_place_plot_text_block(
    fig,
    plot_info_lines,
    font_size=int(plot_info_box_font_size),
    layout=plot_info_box_layout,
)

plot_col, output_col = st.columns([2.2, 1.1])

with plot_col:
    st.subheader(_t("main.plot"))
    st.plotly_chart(fig, use_container_width=True, theme=None)

with output_col:
    st.subheader(_t("main.scientific_output"))
    if fit_model == "exp":
        st.latex(r"\ln(y) = ax + b_{\ln}")
        st.latex(r"y = k\,e^{ax}")
        st.write(f"a_fit = {fit_result.k_fit:.6g}")
        st.write(f"b_ln_fit = {fit_result.l_fit:.6g}")
        if fit_prefactor is not None:
            st.write(f"k_prefactor = {fit_prefactor:.6g}")
    else:
        st.latex(r"y = ax + b")
        st.write(f"a_fit = {fit_result.k_fit:.6g}")
        st.write(f"b_fit = {fit_result.l_fit:.6g}")

    if fit_triangle_slope is not None:
        st.write(_t("scientific.a_triangle_fit", value=f"{fit_triangle_slope:.6g}"))
    else:
        st.write(_t("scientific.a_triangle_fit_not_shown"))

    if error_line_result is not None:
        if error_line_method == "protocol":
            st.caption(_t("scientific.error_method_standard"))
        else:
            st.caption(_t("scientific.error_method_centroid"))
        st.write(f"a_max = {error_line_result.k_max:.6g}")
        st.write(f"a_min = {error_line_result.k_min:.6g}")
        st.write(_t("scientific.delta_a", value=f"{error_line_result.delta_k:.6g}"))
        st.write(_t("scientific.m1_m2", m1=f"{error_line_result.k_max:.6g}", m2=f"{error_line_result.k_min:.6g}"))
        st.write(_t("scientific.delta_m", value=f"{error_line_result.delta_k:.6g}"))
        final_slope = format_final_slope(fit_result.k_fit, error_line_result.delta_k)
        st.write(final_slope)
    else:
        final_slope = None
        st.write(_t("scientific.not_available"))

st.subheader(_t("export.header"))
with st.sidebar:
    st.header(_t("export.sidebar_header"))
    export_base = st.text_input(_t("export.filename_prefix"), value="graphik_plot", key="export_base")
    png_paper = st.selectbox(_t("export.paper_size"), options=["A4", "A5", "Letter"], index=0, key="png_paper")
    png_orientation = st.selectbox(
        _t("export.orientation"),
        options=["portrait", "landscape"],
        index=1,
        key="png_orientation",
        format_func=lambda value: _t(f"orientation.{value}"),
    )
    png_dpi = st.number_input(_t("export.png_dpi"), min_value=72, max_value=600, value=300, step=1, key="png_dpi")
    png_scale = st.number_input(_t("export.png_scale"), min_value=1.0, max_value=4.0, value=1.0, step=0.1, key="png_scale")
    png_word_like = st.checkbox(
        _t("export.word_like_text"),
        value=True,
        key="png_word_like",
    )
    png_text_pt = st.number_input(
        _t("export.target_text_size"),
        min_value=6.0,
        max_value=36.0,
        value=14.0,
        step=0.5,
        disabled=not png_word_like,
        key="png_text_pt",
    )
    png_visual_scale = st.number_input(
        _t("export.extra_visual_scale"),
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.1,
        key="png_visual_scale",
    )
    autoscale_export_axes = st.checkbox(
        _t("export.autoscale"),
        value=True,
        key="autoscale_export_axes",
        help=_t("export.autoscale_help"),
    )

paper_w_mm, paper_h_mm = _paper_size_mm(png_paper)
if png_orientation == "landscape":
    paper_w_mm, paper_h_mm = paper_h_mm, paper_w_mm
# Keep layout sized at base DPI so text remains readable onscreen;
# export with real target canvas size (not only visual upscale).
base_export_dpi = int(png_dpi)
png_width = _mm_to_px(paper_w_mm, base_export_dpi)
png_height = _mm_to_px(paper_h_mm, base_export_dpi)
png_export_scale = float(max(0.2, png_scale))

export_cols = st.columns(5)

with export_cols[0]:
    st.download_button(
        _t("export.download_csv"),
        data=dataframe_to_csv_bytes(analysis_df),
        file_name=f"{export_base}_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )

def _build_normal_export_figure() -> go.Figure:
    export_figure = (
        _autoscale_figure_to_data(fig, x_data, y_data, sigma_y_data, y_axis_type)
        if autoscale_export_axes
        else go.Figure(fig)
    )
    export_figure.update_layout(width=png_width, height=png_height)
    export_figure = _scale_figure_for_export(
        export_figure,
        visual_scale=float(png_visual_scale),
        target_text_pt=float(png_text_pt) if png_word_like else None,
        base_export_dpi=base_export_dpi,
    )
    export_box_font_size = _export_plot_text_font_size(
        preview_figure=fig,
        export_figure=export_figure,
        preview_box_font_size=int(plot_info_box_font_size),
        manual_layout=bool(plot_info_box_layout.manual),
        fallback_font_size=int(annotation_font_size),
    )
    _place_plot_text_block(
        export_figure,
        plot_info_lines,
        font_size=export_box_font_size,
        layout=plot_info_box_layout,
    )
    return export_figure

with export_cols[1]:
    _render_on_demand_image_export(
        cache_key="normal_png",
        prepare_label=_t("export.prepare", format="PNG"),
        spinner_label=_t("export.preparing", format="PNG"),
        download_label=_t("export.download_png", paper=png_paper, orientation=_t(f"orientation.{png_orientation}")),
        file_name=f"{export_base}.png",
        mime="image/png",
        signature=_export_signature(
            fig,
            mode="normal",
            format="png",
            width=png_width,
            height=png_height,
            scale=png_export_scale,
            autoscale_export_axes=bool(autoscale_export_axes),
            y_axis_type=y_axis_type,
            base_export_dpi=base_export_dpi,
            word_like=bool(png_word_like),
            target_text_pt=float(png_text_pt) if png_word_like else None,
            visual_scale=float(png_visual_scale),
            plot_info_lines=plot_info_lines,
            plot_info_box_layout=asdict(plot_info_box_layout),
            plot_info_box_font_size=int(plot_info_box_font_size),
            annotation_font_size=int(annotation_font_size),
        ),
        build_bytes=lambda: figure_to_image_bytes(
            _build_normal_export_figure(),
            "png",
            width=png_width,
            height=png_height,
            scale=png_export_scale,
        ),
        unavailable_message_key="export.png_unavailable",
    )

with export_cols[2]:
    _render_on_demand_image_export(
        cache_key="normal_svg",
        prepare_label=_t("export.prepare", format="SVG"),
        spinner_label=_t("export.preparing", format="SVG"),
        download_label=_t("export.download_svg"),
        file_name=f"{export_base}.svg",
        mime="image/svg+xml",
        signature=_export_signature(
            fig,
            mode="normal",
            format="svg",
            width=png_width,
            height=png_height,
            scale=1.0,
            autoscale_export_axes=bool(autoscale_export_axes),
            y_axis_type=y_axis_type,
            base_export_dpi=base_export_dpi,
            word_like=bool(png_word_like),
            target_text_pt=float(png_text_pt) if png_word_like else None,
            visual_scale=float(png_visual_scale),
            plot_info_lines=plot_info_lines,
            plot_info_box_layout=asdict(plot_info_box_layout),
            plot_info_box_font_size=int(plot_info_box_font_size),
            annotation_font_size=int(annotation_font_size),
        ),
        build_bytes=lambda: figure_to_image_bytes(
            _build_normal_export_figure(),
            "svg",
            width=png_width,
            height=png_height,
            scale=1.0,
        ),
        unavailable_message_key="export.svg_unavailable",
    )

with export_cols[3]:
    _render_on_demand_image_export(
        cache_key="normal_pdf",
        prepare_label=_t("export.prepare", format="PDF"),
        spinner_label=_t("export.preparing", format="PDF"),
        download_label=_t("export.download_pdf", paper=png_paper, orientation=_t(f"orientation.{png_orientation}")),
        file_name=f"{export_base}.pdf",
        mime="application/pdf",
        signature=_export_signature(
            fig,
            mode="normal",
            format="pdf",
            width=png_width,
            height=png_height,
            scale=1.0,
            autoscale_export_axes=bool(autoscale_export_axes),
            y_axis_type=y_axis_type,
            base_export_dpi=base_export_dpi,
            word_like=bool(png_word_like),
            target_text_pt=float(png_text_pt) if png_word_like else None,
            visual_scale=float(png_visual_scale),
            plot_info_lines=plot_info_lines,
            plot_info_box_layout=asdict(plot_info_box_layout),
            plot_info_box_font_size=int(plot_info_box_font_size),
            annotation_font_size=int(annotation_font_size),
        ),
        build_bytes=lambda: figure_to_image_bytes(
            _build_normal_export_figure(),
            "pdf",
            width=png_width,
            height=png_height,
            scale=1.0,
        ),
        unavailable_message_key="export.pdf_unavailable",
    )

summary_text = _build_summary_text(
    raw_df=edited_df,
    analysis_df=analysis_df,
    fit=asdict(fit_result),
    fit_model=fit_model,
    fit_prefactor=fit_prefactor,
    fit_triangle_slope=fit_triangle_slope,
    error=asdict(error_line_result) if error_line_result else None,
    final_slope_text=final_slope,
    lang=str(st.session_state.get("language", "de")),
)

with export_cols[4]:
    st.download_button(
        _t("export.download_summary_md"),
        data=summary_text.encode("utf-8"),
        file_name=f"{export_base}_summary.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.download_button(
    _t("export.download_summary_txt"),
    data=summary_text.encode("utf-8"),
    file_name=f"{export_base}_summary.txt",
    mime="text/plain",
)

st.session_state["_prefs"] = {
    "app_mode": "normal",
    "mapping_mode": mapping_mode,
    "use_zero_error": use_zero_error,
    "use_latex_plot": use_latex_plot,
    "language": st.session_state.get("language", "de"),
    "auto_axis_labels": auto_axis_labels,
    "x_label_input": x_label,
    "y_label_input": y_label,
    "y_scale_mode": y_scale_mode,
    "log_decade_mode": decade_mode if y_axis_type == "log" else "auto",
    "use_separate_fonts": use_separate_font_sizes,
    "global_font_size": int(global_font_size) if not use_separate_font_sizes else int(base_font_size),
    "base_font_size": int(base_font_size),
    "axis_title_font_size": int(axis_title_font_size),
    "tick_font_size": int(tick_font_size),
    "annotation_font_size": int(annotation_font_size),
    "plot_info_box_manual": bool(plot_info_box_layout.manual),
    "plot_info_box_font_size": int(plot_info_box_font_size),
    "plot_info_box_x": float(st.session_state.get("plot_info_box_x", 0.02)),
    "plot_info_box_y": float(st.session_state.get("plot_info_box_y", 0.98)),
    "plot_info_box_width": float(st.session_state.get("plot_info_box_width", 0.46)),
    "plot_info_box_height": float(st.session_state.get("plot_info_box_height", 0.28)),
    "show_grid": show_grid,
    "grid_mode": grid_mode,
    "x_major_divisions": int(x_major_divisions),
    "y_major_divisions": int(y_major_divisions),
    "minor_per_major": int(minor_per_major),
    "marker_size": float(marker_size),
    "error_bar_thickness": float(error_bar_thickness),
    "error_bar_cap_width": float(error_bar_cap_width),
    "connect_points": connect_points,
    "x_tick_decimals": int(x_tick_decimals),
    "y_tick_decimals": int(y_tick_decimals),
    "custom_x_range": st.session_state.get("custom_x_range", False),
    "custom_y_range": st.session_state.get("custom_y_range", False),
    "x_min_user": float(x_range[0]) if x_range is not None else float(np.min(x_data)),
    "x_max_user": float(x_range[1]) if x_range is not None else float(np.max(x_data)),
    "y_min_user": float(y_range[0]) if y_range is not None else float(np.min(y_data)),
    "y_max_user": float(y_range[1]) if y_range is not None else float(np.max(y_data)),
    "fit_model": fit_model,
    "show_fit_line": show_fit_line,
    "show_error_lines": show_error_lines,
    "error_line_mode_widget": error_line_mode_ui,
    "auto_line_labels": auto_line_labels_enabled,
    "fit_label_input": fit_label,
    "fit_color": fit_color,
    "show_fit_slope_label": show_fit_slope_label,
    "show_line_equations_on_plot": show_line_equations_on_plot,
    "show_r2_on_plot": show_r2_on_plot,
    "error_display_mode": error_display_mode,
    "visible_error_lines_linear": list(visible_error_lines_linear),
    "visible_error_lines_exp": list(visible_error_lines_exp),
    "error_color": error_color,
    "error_label_max_input": error_label_max,
    "error_label_min_input": error_label_min,
    "show_error_slope_label": show_error_slope_label,
    "show_fit_triangle": show_fit_triangle,
    "auto_fit_points": auto_fit_points,
    "show_error_triangles": show_error_triangles,
    "triangle_x_decimals": int(triangle_x_decimals),
    "triangle_y_decimals": int(triangle_y_decimals),
    "export_base": export_base,
    "png_paper": png_paper,
    "png_orientation": png_orientation,
    "png_dpi": int(png_dpi),
    "png_scale": float(png_scale),
    "png_word_like": png_word_like,
    "png_text_pt": float(png_text_pt),
    "png_visual_scale": float(png_visual_scale),
    "autoscale_export_axes": autoscale_export_axes,
}

if mapping_mode == "simple":
    st.session_state["_prefs"]["x_col_simple"] = x_col
    st.session_state["_prefs"]["y_col_simple"] = y_simple_col
    st.session_state["_prefs"]["sigma_col_simple"] = sigma_simple_col
else:
    st.session_state["_prefs"]["x_col_adv"] = x_col
    st.session_state["_prefs"]["y_col_adv"] = y_col_selected
    st.session_state["_prefs"]["sigma_y_col_adv"] = sigma_y_col_selected

_save_session_snapshot(st.session_state)
