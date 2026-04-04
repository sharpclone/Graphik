"""Analysis services shared by the Streamlit page controllers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.calculations import (
    centroid_error_lines,
    exponential_regression,
    format_final_slope,
    linear_regression,
    logarithmic_transform_with_uncertainty,
    protocol_endpoint_error_lines,
)
from src.data_io import prepare_measurement_data
from src.export_utils import autoscale_figure_to_data, build_summary_text, place_plot_text_block
from src.geometry import auto_triangle_points, custom_points_from_x
from src.mode_models import (
    NormalAnalysisResult,
    NormalModeConfig,
    PlotContract,
    StatisticsAnalysisResult,
    StatisticsModeConfig,
)
from src.plotting import (
    LineStyle,
    PlotStyle,
    add_exponential_line,
    add_line,
    add_slope_triangle,
    create_base_figure,
    visible_x_range,
)
from src.statistics import DistributionStats, describe_distribution, normal_curve_points
from src.ui_helpers import (
    auto_triangle_delta_symbols,
    format_exponential_equation,
    format_linear_equation,
    to_plot_math_text,
)

TranslateFn = Callable[..., str]


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a dataframe as markdown without requiring the optional tabulate package."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        columns = [str(column) for column in df.columns]
        rows = df.fillna("").astype(str).values.tolist()
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
        return "\n".join([header, separator, *body])
EXP_ERROR_MIN_COLOR = "#ff8c00"
EXP_ERROR_MAX_COLOR = "#ffd700"
EXP_ERROR_MEAN_COLOR = "#1f77b4"


def best_statistics_column(columns: list[str], dataframe: pd.DataFrame) -> str:
    """Pick the column with the most numeric values as default statistics input."""
    if not columns:
        return ""
    counts = {
        col: int(pd.to_numeric(dataframe[col], errors="coerce").notna().sum())
        for col in columns
    }
    return max(counts, key=lambda column: counts[column])


def build_statistics_summary_text(
    *,
    raw_df: pd.DataFrame,
    stats_column: str,
    numeric_values_df: pd.DataFrame,
    stats_result: DistributionStats,
    bins: int,
    normalize_density: bool,
    include_normal_fit: bool,
    translate: TranslateFn,
) -> str:
    """Build the exported markdown summary for statistics mode."""
    lines: list[str] = []
    lines.append(translate("statistics.summary_title"))
    lines.append(translate("summary.generated", timestamp=pd.Timestamp.now().isoformat(timespec="seconds")))
    lines.append("")
    lines.append(translate("statistics.summary_column", column=stats_column))
    lines.append("")
    lines.append(translate("statistics.summary_numeric_values"))
    lines.append(_dataframe_to_markdown(numeric_values_df))
    lines.append("")
    lines.append(translate("statistics.summary_metrics"))
    lines.append(translate("statistics.sample_size", value=str(int(stats_result.count))))
    lines.append(translate("statistics.mean_value", value=f"{stats_result.mean:.6g}"))
    lines.append(translate("statistics.std_value", value=f"{stats_result.std:.6g}"))
    lines.append(translate("statistics.variance_value", value=f"{stats_result.variance:.6g}"))
    lines.append(translate("statistics.median_value", value=f"{stats_result.median:.6g}"))
    lines.append(translate("statistics.min_value", value=f"{stats_result.minimum:.6g}"))
    lines.append(translate("statistics.max_value", value=f"{stats_result.maximum:.6g}"))
    lines.append(translate("statistics.q1_value", value=f"{stats_result.q1:.6g}"))
    lines.append(translate("statistics.q3_value", value=f"{stats_result.q3:.6g}"))
    lines.append("")
    lines.append(translate("statistics.summary_histogram"))
    lines.append(f"bins = {int(bins)}")
    lines.append(f"density = {bool(normalize_density)}")
    lines.append("")
    lines.append(translate("statistics.summary_fit"))
    if include_normal_fit:
        lines.append(translate("statistics.mean_value", value=f"{stats_result.mean:.6g}"))
        lines.append(translate("statistics.std_value", value=f"{stats_result.std:.6g}"))
    else:
        lines.append(translate("statistics.summary_fit_unavailable"))
    lines.append("")
    lines.append(translate("summary.raw_data"))
    lines.append(_dataframe_to_markdown(raw_df))
    return "\n".join(lines)


def analyze_statistics_mode(
    edited_df: pd.DataFrame,
    config: StatisticsModeConfig,
    *,
    translate: TranslateFn,
) -> StatisticsAnalysisResult:
    """Build the complete statistics-mode result model."""
    numeric_series = pd.to_numeric(edited_df[config.stats_column], errors="coerce")
    valid_mask = numeric_series.notna()
    numeric_values = numeric_series[valid_mask].to_numpy(dtype=float)
    dropped_count = int((~valid_mask).sum())
    if numeric_values.size == 0:
        raise ValueError(translate("statistics.no_numeric_values"))
    if numeric_values.size < 2:
        raise ValueError(translate("statistics.not_enough_values"))

    stats_result = describe_distribution(numeric_values)
    constant_data = bool(np.isclose(stats_result.std, 0.0))
    include_normal_fit = bool(config.show_normal_fit and not constant_data)

    rendered_x_label = to_plot_math_text(config.x_label, config.use_math_text)
    rendered_y_label = to_plot_math_text(config.y_label, config.use_math_text)
    histogram_counts, histogram_edges = np.histogram(numeric_values, bins=int(config.bins), density=bool(config.normalize_density))
    bin_widths = np.diff(histogram_edges)
    bin_centers = histogram_edges[:-1] + (bin_widths / 2.0)
    y_top = float(np.max(histogram_counts)) * 1.18 if histogram_counts.size else 1.0
    x_fit = np.array([], dtype=float)
    y_fit = np.array([], dtype=float)
    if include_normal_fit:
        x_fit, y_fit = normal_curve_points(numeric_values, mean=stats_result.mean, std=stats_result.std)
        y_top = max(y_top, float(np.max(y_fit)) * 1.15)

    x_candidates = [float(histogram_edges[0]), float(histogram_edges[-1])]
    if x_fit.size:
        x_candidates.extend([float(np.min(x_fit)), float(np.max(x_fit))])
    x_min_raw = float(min(x_candidates))
    x_max_raw = float(max(x_candidates))
    x_span = x_max_raw - x_min_raw
    x_pad = max(1e-12, x_span * 0.05)
    if x_span <= 0:
        x_pad = max(1.0, abs(x_min_raw) * 0.05)
    x_min_plot = x_min_raw - x_pad
    x_max_plot = x_max_raw + x_pad

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=histogram_counts,
            width=bin_widths,
            marker={"color": config.histogram_color, "line": {"color": config.histogram_color, "width": 1.0}},
            name=to_plot_math_text(translate("statistics.histogram_label"), config.use_math_text),
            opacity=0.72,
            hovertemplate="x = %{x:.6g}<br>y = %{y:.6g}<extra></extra>",
        )
    )
    if include_normal_fit:
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                line={"color": config.fit_color, "width": 2.5},
                name=to_plot_math_text(translate("statistics.normal_fit_label"), config.use_math_text),
            )
        )

    def _add_vertical_marker(x_value: float, label: str, color: str, dash: str, group: str) -> None:
        fig.add_trace(
            go.Scatter(
                x=[x_value, x_value],
                y=[0.0, y_top],
                mode="lines",
                line={"color": color, "width": 1.5, "dash": dash},
                name=to_plot_math_text(label, config.use_math_text),
                legendgroup=group,
                hovertemplate=f"x = {x_value:.6g}<extra></extra>",
            )
        )

    if config.show_mean_line:
        _add_vertical_marker(float(stats_result.mean), translate("statistics.mean_legend"), config.mean_color, "dash", "stats_mean")

    if config.show_std_lines and not constant_data:
        sigma_multiples = [1]
        if config.show_two_sigma:
            sigma_multiples.append(2)
        if config.show_three_sigma:
            sigma_multiples.append(3)
        for multiple in sigma_multiples:
            sigma_label = translate("statistics.sigma_legend", multiple=str(multiple))
            for sign, dash_style, show_legend in ((-1.0, "dot", True), (1.0, "dot", False)):
                marker_x = float(stats_result.mean + sign * multiple * stats_result.std)
                fig.add_trace(
                    go.Scatter(
                        x=[marker_x, marker_x],
                        y=[0.0, y_top],
                        mode="lines",
                        line={"color": config.std_color, "width": 1.5, "dash": dash_style},
                        name=to_plot_math_text(sigma_label, config.use_math_text),
                        legendgroup=f"stats_sigma_{multiple}",
                        showlegend=show_legend,
                        hovertemplate=f"x = {marker_x:.6g}<extra></extra>",
                    )
                )

    fig.update_layout(
        template="plotly_white",
        xaxis_title=rendered_x_label,
        yaxis_title=rendered_y_label,
        font={"size": int(config.base_font_size)},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0, "bgcolor": "rgba(255,255,255,0.88)"},
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
        bargap=0.04,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(
        showgrid=config.show_grid,
        gridcolor="rgba(120,120,120,0.32)",
        tickformat=f".{int(config.x_tick_decimals)}f",
        title_font={"size": int(config.axis_title_font_size)},
        tickfont={"size": int(config.tick_font_size)},
        range=[x_min_plot, x_max_plot],
    )
    fig.update_yaxes(
        showgrid=config.show_grid,
        gridcolor="rgba(120,120,120,0.32)",
        tickformat=f".{int(config.y_tick_decimals)}f",
        title_font={"size": int(config.axis_title_font_size)},
        tickfont={"size": int(config.tick_font_size)},
        range=[0.0, y_top],
    )

    plot_info_lines: list[str] = []
    messages: list[tuple[str, str]] = []
    if config.show_formula_box and include_normal_fit:
        plot_info_lines = [
            translate("statistics.formula_title"),
            r"f(x) = 1/(\sigma sqrt(2\pi)) \cdot exp(-(x-\mu)^2 / (2\sigma^2))",
            f"\\mu = {stats_result.mean:.6g}",
            f"\\sigma = {stats_result.std:.6g}",
        ]
    if not include_normal_fit:
        messages.append(("caption", translate("statistics.summary_fit_unavailable")))

    rendered_plot_lines = tuple(to_plot_math_text(line, config.use_math_text) for line in plot_info_lines)
    plot_info_status = place_plot_text_block(fig, list(rendered_plot_lines), font_size=int(config.plot_info_box.font_size), layout=config.plot_info_box.to_layout())

    return StatisticsAnalysisResult(
        numeric_values=tuple(float(v) for v in numeric_values),
        numeric_values_df=pd.DataFrame({config.stats_column: numeric_values}),
        dropped_count=dropped_count,
        stats_result=stats_result,
        include_normal_fit=include_normal_fit,
        is_constant_data=constant_data,
        plot_contract=PlotContract(
            preview_figure=fig,
            plot_info_lines=rendered_plot_lines,
            plot_info_box=config.plot_info_box,
            annotation_font_size=int(config.annotation_font_size),
            build_export_base_figure=lambda autoscale=False: autoscale_figure_to_data(fig, np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), "linear") if autoscale else go.Figure(fig),
            plot_info_status=plot_info_status,
        ),
        messages=tuple(messages),
    )


def prepare_normal_analysis_dataframe(edited_df: pd.DataFrame, config: NormalModeConfig) -> pd.DataFrame:
    """Validate and normalize the table columns selected for normal-mode plotting."""
    prepared_source_df = edited_df.copy()
    sigma_column = config.sigma_y_column
    if config.use_zero_error:
        prepared_source_df["__sigma_zero__"] = 0.0
        sigma_column = "__sigma_zero__"
    prepared = prepare_measurement_data(
        source_df=prepared_source_df,
        x_col=config.x_column,
        y_col=config.y_column,
        sigma_y_col=sigma_column,
    )
    return prepared.dataframe


def build_normal_analysis_result(
    edited_df: pd.DataFrame,
    config: NormalModeConfig,
    *,
    translate: TranslateFn,
) -> NormalAnalysisResult:
    """Build the complete normal-mode analysis result model."""
    analysis_df = prepare_normal_analysis_dataframe(edited_df, config)
    x_data = analysis_df["x"].to_numpy(dtype=float)
    y_data = analysis_df["y"].to_numpy(dtype=float)
    sigma_y_data = analysis_df["sigma_y"].to_numpy(dtype=float)
    data_x_bounds = (float(np.min(x_data)), float(np.max(x_data)))

    messages: list[tuple[str, str]] = []
    if config.y_axis_type == "log":
        if np.any(y_data <= 0):
            raise ValueError(translate("runtime.semilog_positive"))
        if config.y_range is not None and (config.y_range[0] <= 0 or config.y_range[1] <= 0):
            raise ValueError(translate("runtime.custom_log_positive"))
        if np.any((analysis_df["y"] - analysis_df["sigma_y"]) <= 0):
            messages.append(("info", translate("runtime.lower_error_clipped")))

    fit_prefactor: float | None = None
    try:
        if config.fit_model == "exp":
            fit_result = exponential_regression(x_data, y_data)
            fit_prefactor = float(np.exp(fit_result.l_fit))
            y_for_error, sigma_for_error = logarithmic_transform_with_uncertainty(y_data, sigma_y_data)
        else:
            fit_result = linear_regression(x_data, y_data)
            y_for_error, sigma_for_error = y_data, sigma_y_data
    except ValueError as exc:
        raise ValueError(translate("runtime.fit_error", error=exc)) from exc

    error_line_result = None
    error_line_error = None
    error_line_method = "centroid"
    try:
        if config.error_line_mode == "protocol":
            error_line_result = protocol_endpoint_error_lines(x_data, y_for_error, sigma_for_error)
            error_line_method = "protocol"
        else:
            error_line_result = centroid_error_lines(x_data, y_for_error, sigma_for_error)
            error_line_method = "centroid"
    except ValueError as exc:
        error_line_error = str(exc)
        messages.append(("warning", translate("runtime.error_lines_unavailable", error=exc)))

    x_view_min, x_view_max = (
        (float(config.x_range[0]), float(config.x_range[1])) if config.x_range is not None else (float(np.min(x_data)), float(np.max(x_data)))
    )
    y_view_min, y_view_max = (
        (float(config.y_range[0]), float(config.y_range[1])) if config.y_range is not None else (float(np.min(y_data)), float(np.max(y_data)))
    )

    x_tick0: float | None = None
    y_tick0: float | None = None
    x_major_dtick: float | None = None
    y_major_dtick: float | str | None = None
    show_minor_grid = False
    x_minor_dtick: float | None = None
    y_minor_dtick: float | str | None = None
    if config.show_grid and config.grid_mode in {"manual", "millimetric"}:
        if x_view_max > x_view_min:
            x_tick0 = x_view_min
            x_major_dtick = (x_view_max - x_view_min) / float(max(1, config.x_major_divisions))
            if config.grid_mode == "millimetric":
                show_minor_grid = True
                x_minor_dtick = x_major_dtick / float(max(1, config.minor_per_major))
        if config.y_axis_type == "linear" and y_view_max > y_view_min:
            y_tick0 = y_view_min
            y_major_dtick = (y_view_max - y_view_min) / float(max(1, config.y_major_divisions))
            if config.grid_mode == "millimetric":
                show_minor_grid = True
                y_minor_dtick = y_major_dtick / float(max(1, config.minor_per_major))
        elif config.y_axis_type == "log":
            if config.grid_mode == "millimetric":
                y_major_dtick = "D1"
                show_minor_grid = True
                y_minor_dtick = "D1"
            else:
                y_major_dtick = "D2"

    plot_style = PlotStyle(
        x_label=to_plot_math_text(config.x_label, config.use_math_text),
        y_label=to_plot_math_text(config.y_label, config.use_math_text),
        show_grid=config.show_grid,
        measured_points_label=translate("plot.measured_points"),
        y_axis_type=config.y_axis_type,
        y_log_decades=config.y_log_decades,
        connect_points=config.connect_points,
        marker_size=float(config.marker_size),
        error_bar_thickness=float(config.error_bar_thickness),
        error_bar_cap_width=float(config.error_bar_cap_width),
        x_tick_decimals=int(config.x_tick_decimals),
        y_tick_decimals=int(config.y_tick_decimals),
        base_font_size=int(config.base_font_size),
        axis_title_font_size=int(config.axis_title_font_size),
        tick_font_size=int(config.tick_font_size),
        x_tick0=x_tick0,
        y_tick0=y_tick0,
        x_major_dtick=x_major_dtick,
        y_major_dtick=y_major_dtick,
        show_minor_grid=show_minor_grid,
        x_minor_dtick=x_minor_dtick,
        y_minor_dtick=y_minor_dtick,
        x_range=config.x_range,
        y_range=config.y_range,
    )

    fig = create_base_figure(analysis_df, plot_style)
    try:
        validated_visible_bounds = visible_x_range(x_data, custom=config.x_range)
    except ValueError as exc:
        raise ValueError(translate("runtime.axis_range_error", error=exc)) from exc
    visible_bounds = validated_visible_bounds if config.extrapolate_lines else data_x_bounds

    horizontal_delta_symbol, vertical_delta_symbol = auto_triangle_delta_symbols(config.x_label, config.y_label)
    fit_triangle_slope: float | None = None
    error_triangle_slopes: dict[str, float] = {}
    plot_info_lines: list[str] = []
    fit_label_rendered = to_plot_math_text(config.fit_label, config.use_math_text)
    error_label_max_rendered = to_plot_math_text(config.error_label_max, config.use_math_text)
    error_label_min_rendered = to_plot_math_text(config.error_label_min, config.use_math_text)

    if config.show_fit_line:
        fit_line_label = fit_label_rendered
        if config.show_fit_slope_label:
            fit_line_label = to_plot_math_text(f"{config.fit_label} (a={fit_result.k_fit:.4g})", config.use_math_text)
        if config.fit_model == "exp":
            add_exponential_line(fig, fit_result.k_fit, fit_result.l_fit, visible_bounds, LineStyle(color=config.fit_color, dash="solid", width=2.5, label=fit_line_label))
            if config.show_line_equations_on_plot:
                plot_info_lines.append(format_exponential_equation(config.fit_label, slope_log=fit_result.k_fit, intercept_log=fit_result.l_fit))
        else:
            add_line(fig, fit_result.k_fit, fit_result.l_fit, visible_bounds, LineStyle(color=config.fit_color, dash="solid", width=2.5, label=fit_line_label))
            if config.show_line_equations_on_plot:
                plot_info_lines.append(format_linear_equation(config.fit_label, slope=fit_result.k_fit, intercept=fit_result.l_fit))

        show_fit_triangle = bool(config.show_fit_triangle)
        if show_fit_triangle and config.fit_model == "exp":
            messages.append(("info", translate("runtime.fit_triangle_disabled_exp")))
            show_fit_triangle = False
        if show_fit_triangle:
            if config.auto_fit_points:
                point_a, point_b = auto_triangle_points(
                    slope=fit_result.k_fit,
                    intercept=fit_result.l_fit,
                    x_min=data_x_bounds[0],
                    x_max=data_x_bounds[1],
                    margin_fraction=0.08,
                )
            else:
                try:
                    point_a, point_b = custom_points_from_x(
                        fit_result.k_fit,
                        fit_result.l_fit,
                        x_a=float(config.custom_fit_x_a if config.custom_fit_x_a is not None else data_x_bounds[0]),
                        x_b=float(config.custom_fit_x_b if config.custom_fit_x_b is not None else data_x_bounds[1]),
                    )
                except ValueError as exc:
                    messages.append(("warning", translate("runtime.custom_fit_points_invalid", error=exc)))
                    point_a, point_b = auto_triangle_points(
                        slope=fit_result.k_fit,
                        intercept=fit_result.l_fit,
                        x_min=data_x_bounds[0],
                        x_max=data_x_bounds[1],
                        margin_fraction=0.08,
                    )
            fit_triangle_slope = add_slope_triangle(
                fig,
                a=point_a,
                b=point_b,
                color=config.fit_color,
                label_prefix=fit_label_rendered,
                horizontal_symbol=horizontal_delta_symbol,
                vertical_symbol=vertical_delta_symbol,
                use_latex=config.use_math_text,
                x_decimals=int(config.triangle_x_decimals),
                y_decimals=int(config.triangle_y_decimals),
                font_size=int(config.annotation_font_size),
                annotate=True,
            )

    if config.show_error_lines and error_line_result is not None:
        if config.fit_model == "exp":
            visible_exp = set(config.visible_error_lines_exp)
            a_min = float(error_line_result.k_min)
            a_max = float(error_line_result.k_max)
            b_min = float(error_line_result.l_min)
            b_max = float(error_line_result.l_max)
            a_mean = float((a_min + a_max) / 2.0)
            b_mean = float((b_min + b_max) / 2.0)

            label_min = translate("line.exp.min")
            label_max = translate("line.exp.max")
            label_mean = translate("line.exp.mean")
            if config.show_error_slope_label:
                label_min = f"{label_min} (a={a_min:.4g})"
                label_max = f"{label_max} (a={a_max:.4g})"
                label_mean = f"{label_mean} (a={a_mean:.4g})"
            if "min" in visible_exp:
                add_exponential_line(fig, a_min, b_min, visible_bounds, LineStyle(color=EXP_ERROR_MIN_COLOR, dash="dot", width=2.0, label=label_min))
            if "max" in visible_exp:
                add_exponential_line(fig, a_max, b_max, visible_bounds, LineStyle(color=EXP_ERROR_MAX_COLOR, dash="dash", width=2.0, label=label_max))
            if "mean" in visible_exp:
                add_exponential_line(fig, a_mean, b_mean, visible_bounds, LineStyle(color=EXP_ERROR_MEAN_COLOR, dash="solid", width=2.4, label=label_mean))
            if config.show_line_equations_on_plot:
                if "min" in visible_exp:
                    plot_info_lines.append(format_exponential_equation(translate("line.eq.exp.min"), slope_log=a_min, intercept_log=b_min))
                if "max" in visible_exp:
                    plot_info_lines.append(format_exponential_equation(translate("line.eq.exp.max"), slope_log=a_max, intercept_log=b_max))
                if "mean" in visible_exp:
                    plot_info_lines.append(format_exponential_equation(translate("line.eq.exp.mean"), slope_log=a_mean, intercept_log=b_mean))
            if config.show_error_triangles and visible_exp:
                messages.append(("info", translate("runtime.error_triangles_disabled_exp")))
        else:
            visible_linear = set(config.visible_error_lines_linear)
            if "k_max" in visible_linear:
                label = error_label_max_rendered if not config.show_error_slope_label else to_plot_math_text(f"{config.error_label_max} (a={error_line_result.k_max:.4g})", config.use_math_text)
                add_line(fig, error_line_result.k_max, error_line_result.l_max, visible_bounds, LineStyle(color=config.error_color, dash="dash", width=2.0, label=label))
                if config.show_line_equations_on_plot:
                    plot_info_lines.append(format_linear_equation(config.error_label_max, slope=error_line_result.k_max, intercept=error_line_result.l_max))
            if "k_min" in visible_linear:
                label = error_label_min_rendered if not config.show_error_slope_label else to_plot_math_text(f"{config.error_label_min} (a={error_line_result.k_min:.4g})", config.use_math_text)
                add_line(fig, error_line_result.k_min, error_line_result.l_min, visible_bounds, LineStyle(color=config.error_color, dash="dot", width=2.0, label=label))
                if config.show_line_equations_on_plot:
                    plot_info_lines.append(format_linear_equation(config.error_label_min, slope=error_line_result.k_min, intercept=error_line_result.l_min))
            if config.show_error_triangles:
                x0, x1 = data_x_bounds
                dx = x1 - x0
                if "k_max" in visible_linear:
                    p1_max, p2_max = custom_points_from_x(error_line_result.k_max, error_line_result.l_max, x_a=float(x0 + 0.05 * dx), x_b=float(x0 + 0.95 * dx))
                    a_max_pt, b_max_pt = (p1_max, p2_max) if p1_max.y >= p2_max.y else (p2_max, p1_max)
                    error_triangle_slopes["k_max"] = add_slope_triangle(fig, a=a_max_pt, b=b_max_pt, color=config.error_color, label_prefix="k_max", horizontal_symbol=horizontal_delta_symbol, vertical_symbol=vertical_delta_symbol, use_latex=config.use_math_text, x_decimals=int(config.triangle_x_decimals), y_decimals=int(config.triangle_y_decimals), font_size=int(config.annotation_font_size), annotate=True)
                if "k_min" in visible_linear:
                    p1_min, p2_min = custom_points_from_x(error_line_result.k_min, error_line_result.l_min, x_a=float(x0 + 0.05 * dx), x_b=float(x0 + 0.95 * dx))
                    a_min_pt, b_min_pt = (p1_min, p2_min) if p1_min.y >= p2_min.y else (p2_min, p1_min)
                    error_triangle_slopes["k_min"] = add_slope_triangle(fig, a=a_min_pt, b=b_min_pt, color=config.error_color, label_prefix="k_min", horizontal_symbol=horizontal_delta_symbol, vertical_symbol=vertical_delta_symbol, use_latex=config.use_math_text, x_decimals=int(config.triangle_x_decimals), y_decimals=int(config.triangle_y_decimals), font_size=int(config.annotation_font_size), annotate=True)

    if config.show_r2_on_plot and config.show_fit_line:
        r_squared = float(fit_result.r_value**2)
        plot_info_lines.append(f"R^{{2}}(ln y) = {r_squared:.6g}" if config.fit_model == "exp" else f"R^{{2}} = {r_squared:.6g}")

    rendered_plot_info_lines = tuple(to_plot_math_text(line, config.use_math_text) for line in plot_info_lines)
    plot_info_status = place_plot_text_block(fig, list(rendered_plot_info_lines), font_size=int(config.plot_info_box.font_size), layout=config.plot_info_box.to_layout())

    final_slope: str | None = None
    if error_line_result is not None:
        final_slope = format_final_slope(fit_result.k_fit, error_line_result.delta_k)

    return NormalAnalysisResult(
        analysis_df=analysis_df,
        x_data=tuple(float(v) for v in x_data),
        y_data=tuple(float(v) for v in y_data),
        sigma_y_data=tuple(float(v) for v in sigma_y_data),
        fit_result=fit_result,
        fit_prefactor=fit_prefactor,
        error_line_result=error_line_result,
        error_line_error=error_line_error,
        error_line_method=error_line_method,
        fit_triangle_slope=fit_triangle_slope,
        error_triangle_slopes=error_triangle_slopes,
        final_slope=final_slope,
        plot_contract=PlotContract(
            preview_figure=fig,
            plot_info_lines=rendered_plot_info_lines,
            plot_info_box=config.plot_info_box,
            annotation_font_size=int(config.annotation_font_size),
            build_export_base_figure=lambda autoscale=False: autoscale_figure_to_data(
                fig,
                x_data,
                y_data,
                sigma_y_data,
                config.y_axis_type,
                preserve_x_range=config.x_range is not None,
                preserve_y_range=config.y_range is not None,
            ) if autoscale else go.Figure(fig),
            plot_info_status=plot_info_status,
        ),
        messages=tuple(messages),
    )


def build_normal_summary_text(
    *,
    raw_df: pd.DataFrame,
    analysis_result: NormalAnalysisResult,
    fit_model: str,
    translate: TranslateFn,
) -> str:
    """Build the exported markdown summary for normal mode."""
    return build_summary_text(
        raw_df=raw_df,
        analysis_df=analysis_result.analysis_df,
        fit={
            "k_fit": analysis_result.fit_result.k_fit,
            "l_fit": analysis_result.fit_result.l_fit,
            "r_value": analysis_result.fit_result.r_value,
            "p_value": analysis_result.fit_result.p_value,
            "std_err": analysis_result.fit_result.std_err,
        },
        fit_model=fit_model,
        fit_prefactor=analysis_result.fit_prefactor,
        fit_triangle_slope=analysis_result.fit_triangle_slope,
        error={
            "x_bar": analysis_result.error_line_result.x_bar,
            "y_bar": analysis_result.error_line_result.y_bar,
            "k_min": analysis_result.error_line_result.k_min,
            "k_max": analysis_result.error_line_result.k_max,
            "l_min": analysis_result.error_line_result.l_min,
            "l_max": analysis_result.error_line_result.l_max,
            "delta_k": analysis_result.error_line_result.delta_k,
        }
        if analysis_result.error_line_result
        else None,
        final_slope_text=analysis_result.final_slope,
        lang="de" if translate("app.title") == "Graphik" and translate("settings.header") == "Einstellungen" else "en",
    )
