"""Normal-mode page controller and sidebar schema."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import streamlit as st

from pages.common import (
    render_font_controls,
    render_plot_info_box_controls,
    render_plot_info_box_status,
    render_problem_list,
)
from services.analysis_service import build_normal_analysis_result, build_normal_summary_text
from services.export_service import ExportRequest, build_export_figure, render_export_buttons, render_export_settings
from services.validation_service import collect_normal_mode_problems, has_blocking_problems, suggest_column_mapping
from src.config import DEFAULT_ERROR_LINE_COLOR, DEFAULT_FIT_COLOR
from src.data_io import dataframe_to_csv_bytes
from src.mode_models import NormalModeConfig
from src.ui_helpers import (
    auto_axis_labels,
    auto_line_labels,
    auto_triangle_delta_symbols,
    error_line_help_text,
    fit_line_help_text,
    safe_default_index,
)

NORMAL_PREFIX = "normal."
TranslateFn = Callable[..., str]


def _k(name: str) -> str:
    return f"{NORMAL_PREFIX}{name}"


def render_normal_controls(
    edited_df: pd.DataFrame,
    columns: list[str],
    translate: TranslateFn,
) -> NormalModeConfig:
    """Render the normal-mode sidebar controls and return a typed config object."""
    none_option = "<None>"
    with st.sidebar:
        st.header(translate("mapping.header"))
        suggested_x, suggested_y, suggested_sigma, _ = suggest_column_mapping(columns)
        mapping_mode = st.radio(
            translate("mapping.mode"),
            options=["simple", "advanced"],
            index=0,
            key=_k("mapping_mode"),
            format_func=lambda value: translate(f"mapping_mode.{value}"),
        )
        use_zero_error = bool(st.checkbox(translate("mapping.no_error_column"), value=False, key=_k("use_zero_error")))
        if mapping_mode == "simple":
            st.caption(translate("mapping.simple_caption"))
            x_col = st.selectbox(translate("mapping.x_column"), options=columns, index=safe_default_index(columns, suggested_x or "x"), key=_k("x_col_simple"))
            y_col = st.selectbox(translate("mapping.y_column"), options=columns, index=safe_default_index(columns, suggested_y or "y", fallback=min(1, len(columns) - 1)), key=_k("y_col_simple"))
            if use_zero_error:
                sigma_y_col = none_option
                st.caption(translate("mapping.vertical_unc_disabled"))
            else:
                sigma_y_col = st.selectbox(translate("mapping.error_column"), options=columns, index=safe_default_index(columns, suggested_sigma or "sigma_y", fallback=min(2, len(columns) - 1)), key=_k("sigma_col_simple"))
        else:
            x_col = st.selectbox(translate("mapping.x_column"), options=columns, index=safe_default_index(columns, suggested_x or "x"), key=_k("x_col_adv"))
            y_col = st.selectbox(translate("mapping.y_column"), options=columns, index=safe_default_index(columns, suggested_y or "y", fallback=min(1, len(columns) - 1)), key=_k("y_col_adv"))
            if use_zero_error:
                sigma_y_col = none_option
                st.caption(translate("mapping.vertical_unc_disabled"))
            else:
                sigma_y_col = st.selectbox(translate("mapping.sigma_y_column"), options=columns, index=safe_default_index(columns, suggested_sigma or "sigma_y", fallback=min(2, len(columns) - 1)), key=_k("sigma_y_col_adv"))

        st.header(translate("plot_settings.header"))
        use_math_text = bool(st.checkbox(translate("plot_settings.use_latex"), value=True, key=_k("use_latex_plot"), help=translate("plot_settings.use_latex_help")))
        auto_x_label, auto_y_label = auto_axis_labels(x_column=x_col, y_column=y_col)
        auto_axis_labels_enabled = bool(st.checkbox(translate("plot_settings.auto_axis_labels"), value=True, key=_k("auto_axis_labels")))
        if auto_axis_labels_enabled:
            st.session_state[_k("x_label_input")] = auto_x_label
            st.session_state[_k("y_label_input")] = auto_y_label
        x_label = st.text_input(translate("plot_settings.x_axis_label"), key=_k("x_label_input"), disabled=auto_axis_labels_enabled, help=translate("plot_settings.x_axis_help"))
        y_label = st.text_input(translate("plot_settings.y_axis_label"), key=_k("y_label_input"), disabled=auto_axis_labels_enabled, help=translate("plot_settings.y_axis_help"))
        if auto_axis_labels_enabled:
            x_label = auto_x_label
            y_label = auto_y_label
        st.caption(translate("plot_settings.axis_tip"))

        y_scale_mode = st.selectbox(translate("plot_settings.y_scale"), options=["linear", "log"], format_func=lambda item: translate(f"y_scale.{item}"), index=0, key=_k("y_scale_mode"))
        y_log_decades: int | None = None
        if y_scale_mode == "log":
            decade_mode = st.selectbox(translate("plot_settings.log_decades"), options=["auto", "1", "2"], format_func=lambda item: translate(f"log_decade.{item}"), index=0, key=_k("log_decade_mode"))
            if decade_mode in {"1", "2"}:
                y_log_decades = int(decade_mode)
            st.caption(translate("plot_settings.log_caption"))

        font_settings = render_font_controls(translate, key_prefix=NORMAL_PREFIX)
        plot_info_box = render_plot_info_box_controls(translate, font_settings.annotation_font_size, key_prefix=NORMAL_PREFIX)
        show_grid = bool(st.checkbox(translate("plot_settings.show_grid"), value=True, key=_k("show_grid")))
        grid_mode = st.selectbox(translate("plot_settings.grid_mode"), options=["auto", "manual", "millimetric"], index=0, disabled=not show_grid, key=_k("grid_mode"), format_func=lambda value: translate(f"grid_mode.{value}"))
        x_major_divisions = 10
        y_major_divisions = 10
        minor_per_major = 10
        if show_grid and grid_mode in {"manual", "millimetric"}:
            x_major_divisions = int(st.number_input(translate("plot_settings.x_major_divisions"), min_value=1, max_value=200, value=10, step=1, key=_k("x_major_divisions")))
            if y_scale_mode == "linear":
                y_major_divisions = int(st.number_input(translate("plot_settings.y_major_divisions"), min_value=1, max_value=200, value=10, step=1, key=_k("y_major_divisions")))
        if show_grid and grid_mode == "millimetric":
            if y_scale_mode == "linear":
                minor_per_major = int(st.number_input(translate("plot_settings.minor_per_major"), min_value=2, max_value=50, value=10, step=1, key=_k("minor_per_major")))
            else:
                st.caption(translate("plot_settings.log_minor_caption"))

        marker_size = float(st.number_input(translate("plot_settings.marker_size"), min_value=1.0, max_value=30.0, value=7.0, step=0.5, key=_k("marker_size")))
        error_bar_thickness = float(st.number_input(translate("plot_settings.error_bar_thickness"), min_value=0.5, max_value=10.0, value=1.8, step=0.1, key=_k("error_bar_thickness")))
        error_bar_cap_width = float(st.number_input(translate("plot_settings.error_bar_cap_width"), min_value=0.0, max_value=20.0, value=6.0, step=0.5, key=_k("error_bar_cap_width")))
        connect_points = bool(st.checkbox(translate("plot_settings.connect_points"), value=False, key=_k("connect_points")))
        x_tick_decimals = int(st.number_input(translate("plot_settings.x_decimals"), min_value=0, max_value=8, value=1, step=1, key=_k("x_tick_decimals")))
        y_tick_decimals = int(st.number_input(translate("plot_settings.y_decimals"), min_value=0, max_value=8, value=2, step=1, key=_k("y_tick_decimals")))

        custom_x_range = bool(st.checkbox(translate("plot_settings.custom_x_range"), key=_k("custom_x_range")))
        x_range = None
        if custom_x_range:
            default_x_min = float(pd.to_numeric(edited_df[x_col], errors="coerce").dropna().min())
            default_x_max = float(pd.to_numeric(edited_df[x_col], errors="coerce").dropna().max())
            x_min_user = st.number_input(translate("plot_settings.x_min"), value=default_x_min, format="%.6f", key=_k("x_min_user"))
            x_max_user = st.number_input(translate("plot_settings.x_max"), value=default_x_max, format="%.6f", key=_k("x_max_user"))
            x_range = (float(x_min_user), float(x_max_user))

        custom_y_range = bool(st.checkbox(translate("plot_settings.custom_y_range"), key=_k("custom_y_range")))
        y_range = None
        if custom_y_range:
            y_numeric = pd.to_numeric(edited_df[y_col], errors="coerce").dropna().to_numpy(dtype=float)
            if y_scale_mode == "log":
                positive_y = y_numeric[y_numeric > 0]
                if positive_y.size == 0:
                    st.error(translate("plot_settings.log_positive_required"))
                    st.stop()
                default_y_min = float(np.min(positive_y))
            else:
                default_y_min = float(np.min(y_numeric))
            default_y_max = float(np.max(y_numeric))
            y_min_user = st.number_input(translate("plot_settings.y_min"), value=default_y_min, format="%.6f", key=_k("y_min_user"))
            y_max_user = st.number_input(translate("plot_settings.y_max"), value=default_y_max, format="%.6f", key=_k("y_max_user"))
            y_range = (float(y_min_user), float(y_max_user))

        st.header(translate("lines.header"))
        fit_model = st.selectbox(translate("lines.fit_equation"), options=["linear", "exp"], format_func=lambda item: translate(f"fit_model.{item}"), index=1 if y_scale_mode == "log" else 0, key=_k("fit_model"), help=translate("fit_model.help"))
        selected_y = pd.to_numeric(edited_df[y_col], errors="coerce").dropna().to_numpy(dtype=float)
        if fit_model == "exp" and selected_y.size and np.any(selected_y <= 0):
            st.caption(translate("validation.exp_unavailable_detail"))
        if y_scale_mode == "log" and selected_y.size and np.any(selected_y <= 0):
            st.caption(translate("validation.log_axis_detail"))
        current_error_mode = str(st.session_state.get(_k("error_line_mode_widget"), "protocol"))
        show_fit_line = bool(st.checkbox(translate("lines.show_fit_line"), value=True, key=_k("show_fit_line"), help=fit_line_help_text(fit_model, lang=str(st.session_state.get("language", "de")))))
        show_error_lines = bool(st.checkbox(translate("lines.show_error_lines"), value=True, key=_k("show_error_lines"), help=error_line_help_text(fit_model, current_error_mode, lang=str(st.session_state.get("language", "de")))))
        extrapolate_lines = bool(
            st.checkbox(
                translate("lines.extrapolate_lines"),
                value=True,
                key=_k("extrapolate_lines"),
                help=translate("lines.extrapolate_lines_help"),
            )
        )
        error_line_mode = st.selectbox(translate("lines.error_method"), options=["protocol", "centroid"], format_func=lambda item: translate(f"error_method.{item}"), index=0, key=_k("error_line_mode_widget"), help=translate("error_method.help"))
        auto_line_labels_enabled = bool(st.checkbox(translate("lines.auto_line_labels"), value=True, key=_k("auto_line_labels")))
        fit_label_auto, error_label_max_auto, error_label_min_auto = auto_line_labels(error_line_mode, lang=str(st.session_state.get("language", "de")))
        if auto_line_labels_enabled:
            st.session_state[_k("fit_label_input")] = fit_label_auto
            st.session_state[_k("error_label_max_input")] = error_label_max_auto
            st.session_state[_k("error_label_min_input")] = error_label_min_auto
        fit_label = st.text_input(translate("lines.fit_label"), key=_k("fit_label_input"), disabled=auto_line_labels_enabled, help=translate("lines.label_help"))
        if auto_line_labels_enabled:
            fit_label = fit_label_auto
        fit_color = st.color_picker(translate("lines.fit_color"), value=DEFAULT_FIT_COLOR, key=_k("fit_color"))
        show_fit_slope_label = bool(st.checkbox(translate("lines.show_fit_slope_label"), value=True, key=_k("show_fit_slope_label")))
        show_line_equations_on_plot = bool(st.checkbox(translate("lines.show_line_equations"), value=False, key=_k("show_line_equations_on_plot")))
        show_r2_on_plot = bool(st.checkbox(translate("lines.show_r2"), value=False, key=_k("show_r2_on_plot")))
        visible_error_lines_linear: tuple[str, ...] = tuple()
        visible_error_lines_exp: tuple[str, ...] = tuple()
        error_color = DEFAULT_ERROR_LINE_COLOR
        if fit_model == "exp":
            st.caption(translate("lines.exp_colors_caption"))
            exp_options = ["min", "max", "mean"]
            exp_default_raw = st.session_state.get(_k("visible_error_lines_exp"), ["max"])
            exp_default = [item for item in exp_default_raw if item in exp_options] or ["max"]
            visible_error_lines_exp = tuple(st.multiselect(translate("lines.visible_exp_error_lines"), options=exp_options, default=exp_default, key=_k("visible_error_lines_exp"), format_func=lambda item: translate(f"exp_error.{item}")))
        else:
            linear_options = ["k_max", "k_min"]
            linear_default_raw = st.session_state.get(_k("visible_error_lines_linear"), ["k_max"])
            linear_default = [item for item in linear_default_raw if item in linear_options] or ["k_max"]
            visible_error_lines_linear = tuple(st.multiselect(translate("lines.visible_linear_error_lines"), options=linear_options, default=linear_default, key=_k("visible_error_lines_linear"), format_func=lambda item: translate(f"linear_error.{item}")))
            error_color = st.color_picker(translate("lines.error_color"), value=DEFAULT_ERROR_LINE_COLOR, key=_k("error_color"))
        error_label_max = st.text_input(translate("lines.error_label_max"), key=_k("error_label_max_input"), disabled=auto_line_labels_enabled, help=translate("lines.label_help"))
        error_label_min = st.text_input(translate("lines.error_label_min"), key=_k("error_label_min_input"), disabled=auto_line_labels_enabled, help=translate("lines.label_help"))
        if auto_line_labels_enabled:
            error_label_max = error_label_max_auto
            error_label_min = error_label_min_auto
        show_error_slope_label = bool(st.checkbox(translate("lines.show_error_slope_label"), value=True, key=_k("show_error_slope_label")))

        st.header(translate("triangles.header"))
        show_fit_triangle = bool(st.checkbox(translate("triangles.show_fit"), value=False, key=_k("show_fit_triangle")))
        auto_fit_points = bool(st.checkbox(translate("triangles.auto_points"), value=True, key=_k("auto_fit_points")))
        dx_symbol, dy_symbol = auto_triangle_delta_symbols(x_label, y_label)
        st.caption(translate("triangles.auto_labels", dx=dx_symbol, dy=dy_symbol))
        custom_fit_x_a = None
        custom_fit_x_b = None
        if not auto_fit_points:
            x_numeric = pd.to_numeric(edited_df[x_col], errors="coerce").dropna().to_numpy(dtype=float)
            x_min_default = float(np.min(x_numeric)) if x_numeric.size else 0.0
            x_max_default = float(np.max(x_numeric)) if x_numeric.size else 1.0
            st.write(translate("runtime.custom_ab_caption"))
            custom_fit_x_a = float(st.number_input("A_x", value=float(x_min_default + 0.15 * (x_max_default - x_min_default)), format="%.6f", key=_k("fit_custom_ax")))
            custom_fit_x_b = float(st.number_input("B_x", value=float(x_min_default + 0.85 * (x_max_default - x_min_default)), format="%.6f", key=_k("fit_custom_bx")))
        show_error_triangles = bool(st.checkbox(translate("triangles.show_error"), value=False, key=_k("show_error_triangles")))
        triangle_x_decimals = int(st.number_input(translate("triangles.dx_decimals"), min_value=0, max_value=8, value=1, step=1, key=_k("triangle_x_decimals")))
        triangle_y_decimals = int(st.number_input(translate("triangles.dy_decimals"), min_value=0, max_value=8, value=2, step=1, key=_k("triangle_y_decimals")))

    return NormalModeConfig(
        mapping_mode=mapping_mode,
        use_zero_error=use_zero_error,
        x_column=x_col,
        y_column=y_col,
        sigma_y_column=sigma_y_col,
        use_math_text=use_math_text,
        auto_axis_labels=auto_axis_labels_enabled,
        x_label=x_label,
        y_label=y_label,
        y_axis_type=y_scale_mode,
        y_log_decades=y_log_decades,
        base_font_size=font_settings.base_font_size,
        axis_title_font_size=font_settings.axis_title_font_size,
        tick_font_size=font_settings.tick_font_size,
        annotation_font_size=font_settings.annotation_font_size,
        plot_info_box=plot_info_box,
        show_grid=show_grid,
        grid_mode=grid_mode,
        x_major_divisions=x_major_divisions,
        y_major_divisions=y_major_divisions,
        minor_per_major=minor_per_major,
        marker_size=marker_size,
        error_bar_thickness=error_bar_thickness,
        error_bar_cap_width=error_bar_cap_width,
        connect_points=connect_points,
        x_tick_decimals=x_tick_decimals,
        y_tick_decimals=y_tick_decimals,
        x_range=x_range,
        y_range=y_range,
        fit_model=fit_model,
        show_fit_line=show_fit_line,
        show_error_lines=show_error_lines,
        extrapolate_lines=extrapolate_lines,
        error_line_mode=error_line_mode,
        fit_label=fit_label,
        fit_color=fit_color,
        show_fit_slope_label=show_fit_slope_label,
        show_line_equations_on_plot=show_line_equations_on_plot,
        show_r2_on_plot=show_r2_on_plot,
        visible_error_lines_linear=visible_error_lines_linear,
        visible_error_lines_exp=visible_error_lines_exp,
        error_color=error_color,
        error_label_max=error_label_max,
        error_label_min=error_label_min,
        show_error_slope_label=show_error_slope_label,
        show_fit_triangle=show_fit_triangle,
        auto_fit_points=auto_fit_points,
        custom_fit_x_a=custom_fit_x_a,
        custom_fit_x_b=custom_fit_x_b,
        show_error_triangles=show_error_triangles,
        triangle_x_decimals=triangle_x_decimals,
        triangle_y_decimals=triangle_y_decimals,
        auto_line_labels_enabled=auto_line_labels_enabled,
    )


def render_normal_mode(
    edited_df: pd.DataFrame,
    columns: list[str],
    translate: TranslateFn,
) -> dict[str, object]:
    """Render the full normal mode and return persisted preferences."""
    config = render_normal_controls(edited_df, columns, translate)
    problems = collect_normal_mode_problems(edited_df, config, translate=translate)
    render_problem_list(problems, translate)
    if has_blocking_problems(problems):
        st.stop()

    try:
        result = build_normal_analysis_result(edited_df, config, translate=translate)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    for level, message in result.messages:
        getattr(st, level)(message)

    plot_col, output_col = st.columns([2.2, 1.1])
    with plot_col:
        st.subheader(translate("main.plot"))
        st.plotly_chart(result.plot_contract.preview_figure, use_container_width=True, theme=None)
        render_plot_info_box_status(result.plot_contract.plot_info_status, translate)

    with output_col:
        st.subheader(translate("main.scientific_output"))
        if config.fit_model == "exp":
            st.latex(r"\ln(y) = ax + b_{\ln}")
            st.latex(r"y = k\,e^{ax}")
            st.write(f"a_fit = {result.fit_result.k_fit:.6g}")
            st.write(f"b_ln_fit = {result.fit_result.l_fit:.6g}")
            if result.fit_prefactor is not None:
                st.write(f"k_prefactor = {result.fit_prefactor:.6g}")
        else:
            st.latex(r"y = ax + b")
            st.write(f"a_fit = {result.fit_result.k_fit:.6g}")
            st.write(f"b_fit = {result.fit_result.l_fit:.6g}")
        if result.fit_triangle_slope is not None:
            st.write(translate("scientific.a_triangle_fit", value=f"{result.fit_triangle_slope:.6g}"))
        else:
            st.write(translate("scientific.a_triangle_fit_not_shown"))
        if result.error_line_result is not None:
            st.caption(translate("scientific.error_method_standard") if result.error_line_method == "protocol" else translate("scientific.error_method_centroid"))
            st.write(f"a_max = {result.error_line_result.k_max:.6g}")
            st.write(f"a_min = {result.error_line_result.k_min:.6g}")
            st.write(translate("scientific.delta_a", value=f"{result.error_line_result.delta_k:.6g}"))
            st.write(translate("scientific.m1_m2", m1=f"{result.error_line_result.k_max:.6g}", m2=f"{result.error_line_result.k_min:.6g}"))
            st.write(translate("scientific.delta_m", value=f"{result.error_line_result.delta_k:.6g}"))
            if result.final_slope is not None:
                st.write(result.final_slope)
        else:
            st.write(translate("scientific.not_available"))

    st.subheader(translate("export.header"))
    with st.sidebar:
        export_config = render_export_settings(
            translate=translate,
            default_base_name="graphik_plot",
            include_autoscale=True,
            key_prefix=NORMAL_PREFIX,
        )
    export_cols = st.columns(5)
    with export_cols[0]:
        st.download_button(
            translate("export.download_csv"),
            data=dataframe_to_csv_bytes(result.analysis_df),
            file_name=f"{export_config.base_name}_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )
    render_export_buttons(
        export_cols[1:4],
        ExportRequest(
            mode="normal",
            cache_prefix="normal",
            preview_figure=result.plot_contract.preview_figure,
            config=export_config,
            plot_info_lines=result.plot_contract.plot_info_lines,
            plot_info_box_layout=result.plot_contract.plot_info_box.to_layout(),
            plot_info_box_font_size=result.plot_contract.plot_info_box.font_size,
            annotation_font_size=result.plot_contract.annotation_font_size,
            signature_payload={
                "fit_model": config.fit_model,
                "y_axis_type": config.y_axis_type,
                "use_math_text": bool(config.use_math_text),
                "show_fit_line": bool(config.show_fit_line),
                "show_error_lines": bool(config.show_error_lines),
            },
        ),
        translate=translate,
        build_base_figure=lambda: result.plot_contract.build_export_base_figure(export_config.autoscale_axes),
    )
    summary_text = build_normal_summary_text(
        raw_df=edited_df,
        analysis_result=result,
        fit_model=config.fit_model,
        translate=translate,
    )
    if export_config.show_clean_export_preview:
        try:
            clean_preview = build_export_figure(
                ExportRequest(
                    mode="normal",
                    cache_prefix="normal_preview",
                    preview_figure=result.plot_contract.preview_figure,
                    config=export_config,
                    plot_info_lines=result.plot_contract.plot_info_lines,
                    plot_info_box_layout=result.plot_contract.plot_info_box.to_layout(),
                    plot_info_box_font_size=result.plot_contract.plot_info_box.font_size,
                    annotation_font_size=result.plot_contract.annotation_font_size,
                    signature_payload={
                        "fit_model": config.fit_model,
                        "preview_kind": "clean_export",
                    },
                ),
                lambda: result.plot_contract.build_export_base_figure(export_config.autoscale_axes),
            )
            with st.expander(translate("export.clean_preview_header"), expanded=False):
                st.plotly_chart(clean_preview, use_container_width=True, theme=None)
        except Exception as exc:  # pragma: no cover - preview convenience boundary
            st.info(translate("export.clean_preview_failed", error=exc))
    with export_cols[4]:
        st.download_button(
            translate("export.download_summary_md"),
            data=summary_text.encode("utf-8"),
            file_name=f"{export_config.base_name}_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )
    st.download_button(
        translate("export.download_summary_txt"),
        data=summary_text.encode("utf-8"),
        file_name=f"{export_config.base_name}_summary.txt",
        mime="text/plain",
    )

    return {
        "app_mode": "normal",
        _k("mapping_mode"): config.mapping_mode,
        _k("use_zero_error"): config.use_zero_error,
        _k("use_latex_plot"): config.use_math_text,
        "language": st.session_state.get("language", "de"),
        _k("auto_axis_labels"): config.auto_axis_labels,
        _k("x_label_input"): config.x_label,
        _k("y_label_input"): config.y_label,
        _k("y_scale_mode"): config.y_axis_type,
        _k("log_decade_mode"): str(config.y_log_decades) if config.y_log_decades is not None else "auto",
        _k("use_separate_fonts"): bool(st.session_state.get(_k("use_separate_fonts"), False)),
        _k("global_font_size"): int(st.session_state.get(_k("global_font_size"), config.base_font_size)),
        _k("base_font_size"): int(config.base_font_size),
        _k("axis_title_font_size"): int(config.axis_title_font_size),
        _k("tick_font_size"): int(config.tick_font_size),
        _k("annotation_font_size"): int(config.annotation_font_size),
        _k("plot_info_box_manual"): bool(config.plot_info_box.manual),
        _k("plot_info_box_font_size"): int(config.plot_info_box.font_size),
        _k("plot_info_box_x"): float(st.session_state.get(_k("plot_info_box_x"), 0.02)),
        _k("plot_info_box_y"): float(st.session_state.get(_k("plot_info_box_y"), 0.98)),
        _k("plot_info_box_width"): float(st.session_state.get(_k("plot_info_box_width"), 0.46)),
        _k("plot_info_box_height"): float(st.session_state.get(_k("plot_info_box_height"), 0.28)),
        _k("show_grid"): bool(config.show_grid),
        _k("grid_mode"): config.grid_mode,
        _k("x_major_divisions"): int(config.x_major_divisions),
        _k("y_major_divisions"): int(config.y_major_divisions),
        _k("minor_per_major"): int(config.minor_per_major),
        _k("marker_size"): float(config.marker_size),
        _k("error_bar_thickness"): float(config.error_bar_thickness),
        _k("error_bar_cap_width"): float(config.error_bar_cap_width),
        _k("connect_points"): config.connect_points,
        _k("x_tick_decimals"): int(config.x_tick_decimals),
        _k("y_tick_decimals"): int(config.y_tick_decimals),
        _k("custom_x_range"): st.session_state.get(_k("custom_x_range"), False),
        _k("custom_y_range"): st.session_state.get(_k("custom_y_range"), False),
        _k("x_min_user"): float(config.x_range[0]) if config.x_range is not None else float(min(result.x_data)),
        _k("x_max_user"): float(config.x_range[1]) if config.x_range is not None else float(max(result.x_data)),
        _k("y_min_user"): float(config.y_range[0]) if config.y_range is not None else float(min(result.y_data)),
        _k("y_max_user"): float(config.y_range[1]) if config.y_range is not None else float(max(result.y_data)),
        _k("fit_model"): config.fit_model,
        _k("show_fit_line"): config.show_fit_line,
        _k("show_error_lines"): config.show_error_lines,
        _k("error_line_mode_widget"): config.error_line_mode,
        _k("auto_line_labels"): config.auto_line_labels_enabled,
        _k("fit_label_input"): config.fit_label,
        _k("fit_color"): config.fit_color,
        _k("show_fit_slope_label"): config.show_fit_slope_label,
        _k("show_line_equations_on_plot"): config.show_line_equations_on_plot,
        _k("show_r2_on_plot"): config.show_r2_on_plot,
        _k("visible_error_lines_exp"): list(config.visible_error_lines_exp),
        _k("visible_error_lines_linear"): list(config.visible_error_lines_linear),
        _k("error_color"): config.error_color,
        _k("error_label_max_input"): config.error_label_max,
        _k("error_label_min_input"): config.error_label_min,
        _k("show_error_slope_label"): config.show_error_slope_label,
        _k("show_fit_triangle"): config.show_fit_triangle,
        _k("auto_fit_points"): config.auto_fit_points,
        _k("show_error_triangles"): config.show_error_triangles,
        _k("triangle_x_decimals"): int(config.triangle_x_decimals),
        _k("triangle_y_decimals"): int(config.triangle_y_decimals),
        _k("export_base"): export_config.base_name,
        _k("export_preset"): export_config.preset_name,
        _k("show_clean_export_preview"): bool(export_config.show_clean_export_preview),
        _k("png_paper"): export_config.paper,
        _k("png_orientation"): export_config.orientation,
        _k("png_dpi"): int(export_config.dpi),
        _k("png_scale"): float(export_config.raster_scale),
        _k("png_word_like"): bool(export_config.target_text_pt is not None),
        _k("png_text_pt"): float(st.session_state.get(_k("png_text_pt"), export_config.target_text_pt or 14.0)),
        _k("png_visual_scale"): float(export_config.visual_scale),
        _k("autoscale_export_axes"): export_config.autoscale_axes,
    }
