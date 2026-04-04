"""Statistics-mode page controller and sidebar schema."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from pages.common import (
    render_font_controls,
    render_plot_info_box_controls,
    render_plot_info_box_status,
    render_problem_list,
)
from services.analysis_service import analyze_statistics_mode, best_statistics_column, build_statistics_summary_text
from services.export_service import ExportRequest, build_export_figure, render_export_buttons, render_export_settings
from services.validation_service import collect_statistics_mode_problems, has_blocking_problems
from src.data_io import dataframe_to_csv_bytes
from src.mode_models import StatisticsModeConfig
from src.ui_helpers import safe_default_index

TranslateFn = Callable[..., str]
STATISTICS_PREFIX = "stats."


def _k(name: str) -> str:
    return f"{STATISTICS_PREFIX}{name}"


def render_statistics_controls(
    edited_df: pd.DataFrame,
    columns: list[str],
    translate: TranslateFn,
) -> StatisticsModeConfig:
    """Render the statistics sidebar controls and return a typed config object."""
    default_stats_column = best_statistics_column(columns, edited_df)
    st.header(translate("statistics.header"))
    st.caption(translate("statistics.caption"))
    stats_column = st.selectbox(
        translate("statistics.column"),
        options=columns,
        index=safe_default_index(columns, default_stats_column),
        key=_k("stats_column"),
        help=translate("statistics.column_help"),
    )
    bins = int(st.slider(translate("statistics.bins"), min_value=5, max_value=60, value=16, step=1, key=_k("stats_bins")))
    normalize_density = bool(st.checkbox(translate("statistics.normalize_density"), value=False, key=_k("stats_normalize_density")))
    use_math_text = bool(st.checkbox(translate("plot_settings.use_latex"), value=True, key=_k("use_latex_plot"), help=translate("plot_settings.use_latex_help")))
    auto_axis_labels = bool(st.checkbox(translate("statistics.auto_axis_labels"), value=True, key=_k("auto_axis_labels")))
    default_x_label = stats_column
    default_y_label = translate("statistics.default_y_label_density") if normalize_density else translate("statistics.default_y_label_count")
    if auto_axis_labels:
        st.session_state[_k("x_label_input")] = default_x_label
        st.session_state[_k("y_label_input")] = default_y_label
    x_label = st.text_input(translate("statistics.x_axis_label"), key=_k("x_label_input"), disabled=auto_axis_labels)
    y_label = st.text_input(translate("statistics.y_axis_label"), key=_k("y_label_input"), disabled=auto_axis_labels)
    if auto_axis_labels:
        x_label = default_x_label
        y_label = default_y_label
    font_settings = render_font_controls(translate, key_prefix=STATISTICS_PREFIX)
    plot_info_box = render_plot_info_box_controls(translate, font_settings.annotation_font_size, key_prefix=STATISTICS_PREFIX)
    show_grid = bool(st.checkbox(translate("plot_settings.show_grid"), value=True, key=_k("show_grid")))
    x_tick_decimals = int(st.number_input(translate("plot_settings.x_decimals"), min_value=0, max_value=8, value=2, step=1, key=_k("x_tick_decimals")))
    y_tick_decimals = int(st.number_input(translate("plot_settings.y_decimals"), min_value=0, max_value=8, value=2, step=1, key=_k("y_tick_decimals")))
    numeric_values = pd.to_numeric(edited_df[stats_column], errors="coerce").dropna().to_numpy(dtype=float)
    normal_fit_available = bool(numeric_values.size >= 2 and not (numeric_values.size and (numeric_values == numeric_values[0]).all()))
    if not normal_fit_available:
        st.caption(translate("validation.normal_fit_unavailable_detail"))
    show_normal_fit = bool(st.checkbox(translate("statistics.show_normal_fit"), value=True, key=_k("show_normal_fit"), disabled=not normal_fit_available, help=None if normal_fit_available else translate("validation.normal_fit_unavailable_detail")))
    show_formula_box = bool(st.checkbox(translate("statistics.show_formula_box"), value=True, key=_k("show_formula_box")))
    show_mean_line = bool(st.checkbox(translate("statistics.show_mean_line"), value=True, key=_k("show_mean_line")))
    show_std_lines = bool(st.checkbox(translate("statistics.show_std_lines"), value=True, key=_k("show_std_lines")))
    show_two_sigma = bool(st.checkbox(translate("statistics.show_two_sigma"), value=False, key=_k("show_two_sigma")))
    show_three_sigma = bool(st.checkbox(translate("statistics.show_three_sigma"), value=False, key=_k("show_three_sigma")))
    histogram_color = st.color_picker(translate("statistics.histogram_color"), value="#7aa6ff", key=_k("histogram_color"))
    fit_color = st.color_picker(translate("statistics.fit_color"), value="#d62728", key=_k("fit_color"))
    mean_color = st.color_picker(translate("statistics.mean_color"), value="#222222", key=_k("mean_color"))
    std_color = st.color_picker(translate("statistics.std_color"), value="#2ca02c", key=_k("std_color"))
    return StatisticsModeConfig(
        stats_column=stats_column,
        bins=bins,
        normalize_density=normalize_density,
        use_math_text=use_math_text,
        auto_axis_labels=auto_axis_labels,
        x_label=x_label,
        y_label=y_label,
        base_font_size=font_settings.base_font_size,
        axis_title_font_size=font_settings.axis_title_font_size,
        tick_font_size=font_settings.tick_font_size,
        annotation_font_size=font_settings.annotation_font_size,
        plot_info_box=plot_info_box,
        show_grid=show_grid,
        x_tick_decimals=x_tick_decimals,
        y_tick_decimals=y_tick_decimals,
        show_normal_fit=show_normal_fit,
        show_formula_box=show_formula_box,
        show_mean_line=show_mean_line,
        show_std_lines=show_std_lines,
        show_two_sigma=show_two_sigma,
        show_three_sigma=show_three_sigma,
        histogram_color=histogram_color,
        fit_color=fit_color,
        mean_color=mean_color,
        std_color=std_color,
    )


def render_statistics_mode(
    edited_df: pd.DataFrame,
    columns: list[str],
    translate: TranslateFn,
) -> dict[str, object]:
    """Render the full statistics mode and return persisted preferences."""
    with st.sidebar:
        config = render_statistics_controls(edited_df, columns, translate)

    problems = collect_statistics_mode_problems(edited_df, config, translate=translate)
    render_problem_list(problems, translate)
    if has_blocking_problems(problems):
        st.stop()

    try:
        result = analyze_statistics_mode(edited_df, config, translate=translate)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if result.dropped_count > 0:
        st.info(translate("statistics.numeric_rows_used", used=str(int(len(result.numeric_values))), dropped=str(result.dropped_count)))
    if result.is_constant_data:
        st.info(translate("statistics.fit_unavailable_constant"))

    plot_col, output_col = st.columns([2.2, 1.1])
    with plot_col:
        st.subheader(translate("main.plot"))
        st.plotly_chart(result.plot_contract.preview_figure, use_container_width=True, theme=None)
        render_plot_info_box_status(result.plot_contract.plot_info_status, translate)

    with output_col:
        st.subheader(translate("statistics.output_header"))
        st.latex(r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)")
        st.write(translate("statistics.sample_size", value=str(int(result.stats_result.count))))
        st.latex(rf"\mu = {result.stats_result.mean:.6g}")
        st.latex(rf"\sigma = {result.stats_result.std:.6g}")
        st.write(translate("statistics.variance_value", value=f"{result.stats_result.variance:.6g}"))
        st.write(translate("statistics.median_value", value=f"{result.stats_result.median:.6g}"))
        st.write(translate("statistics.min_value", value=f"{result.stats_result.minimum:.6g}"))
        st.write(translate("statistics.max_value", value=f"{result.stats_result.maximum:.6g}"))
        st.write(translate("statistics.q1_value", value=f"{result.stats_result.q1:.6g}"))
        st.write(translate("statistics.q3_value", value=f"{result.stats_result.q3:.6g}"))
        for level, message in result.messages:
            getattr(st, level)(message)

    st.subheader(translate("export.header"))
    with st.sidebar:
        export_config = render_export_settings(
            translate=translate,
            default_base_name="statistics_plot",
            include_autoscale=True,
            key_prefix=STATISTICS_PREFIX,
        )

    summary_text = build_statistics_summary_text(
        raw_df=edited_df,
        stats_column=config.stats_column,
        numeric_values_df=result.numeric_values_df,
        stats_result=result.stats_result,
        bins=config.bins,
        normalize_density=config.normalize_density,
        include_normal_fit=result.include_normal_fit,
        translate=translate,
    )
    if export_config.show_clean_export_preview:
        try:
            clean_preview = build_export_figure(
                ExportRequest(
                    mode="statistics",
                    cache_prefix="statistics_preview",
                    preview_figure=result.plot_contract.preview_figure,
                    config=export_config,
                    plot_info_lines=result.plot_contract.plot_info_lines,
                    plot_info_box_layout=result.plot_contract.plot_info_box.to_layout(),
                    plot_info_box_font_size=result.plot_contract.plot_info_box.font_size,
                    annotation_font_size=result.plot_contract.annotation_font_size,
                    signature_payload={
                        "stats_column": config.stats_column,
                        "preview_kind": "clean_export",
                    },
                ),
                lambda: result.plot_contract.build_export_base_figure(export_config.autoscale_axes),
            )
            with st.expander(translate("export.clean_preview_header"), expanded=False):
                st.plotly_chart(clean_preview, use_container_width=True, theme=None)
        except Exception as exc:  # pragma: no cover - preview convenience boundary
            st.info(translate("export.clean_preview_failed", error=exc))
    export_cols = st.columns(5)
    with export_cols[0]:
        st.download_button(
            translate("statistics.export_values_csv"),
            data=dataframe_to_csv_bytes(result.numeric_values_df),
            file_name=f"{export_config.base_name}_numeric_values.csv",
            mime="text/csv",
            use_container_width=True,
        )
    render_export_buttons(
        export_cols[1:4],
        ExportRequest(
            mode="statistics",
            cache_prefix="statistics",
            preview_figure=result.plot_contract.preview_figure,
            config=export_config,
            plot_info_lines=result.plot_contract.plot_info_lines,
            plot_info_box_layout=result.plot_contract.plot_info_box.to_layout(),
            plot_info_box_font_size=result.plot_contract.plot_info_box.font_size,
            annotation_font_size=result.plot_contract.annotation_font_size,
            signature_payload={
                "stats_column": config.stats_column,
                "normalize_density": bool(config.normalize_density),
                "include_normal_fit": bool(result.include_normal_fit),
                "use_math_text": bool(config.use_math_text),
            },
        ),
        translate=translate,
        build_base_figure=lambda: result.plot_contract.build_export_base_figure(export_config.autoscale_axes),
    )
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
        "app_mode": "statistics",
        "language": st.session_state.get("language", "de"),
        _k("stats_column"): config.stats_column,
        _k("stats_bins"): int(config.bins),
        _k("stats_normalize_density"): bool(config.normalize_density),
        _k("use_latex_plot"): bool(config.use_math_text),
        _k("auto_axis_labels"): bool(config.auto_axis_labels),
        _k("x_label_input"): config.x_label,
        _k("y_label_input"): config.y_label,
        _k("use_separate_fonts"): bool(st.session_state.get(_k("use_separate_fonts"), False)),
        _k("global_font_size"): int(st.session_state.get(_k("global_font_size"), config.base_font_size)),
        _k("base_font_size"): int(config.base_font_size),
        _k("axis_title_font_size"): int(config.axis_title_font_size),
        _k("tick_font_size"): int(config.tick_font_size),
        _k("annotation_font_size"): int(config.annotation_font_size),
        _k("show_grid"): bool(config.show_grid),
        _k("x_tick_decimals"): int(config.x_tick_decimals),
        _k("y_tick_decimals"): int(config.y_tick_decimals),
        _k("show_normal_fit"): bool(config.show_normal_fit),
        _k("show_formula_box"): bool(config.show_formula_box),
        _k("show_mean_line"): bool(config.show_mean_line),
        _k("show_std_lines"): bool(config.show_std_lines),
        _k("show_two_sigma"): bool(config.show_two_sigma),
        _k("show_three_sigma"): bool(config.show_three_sigma),
        _k("histogram_color"): config.histogram_color,
        _k("fit_color"): config.fit_color,
        _k("mean_color"): config.mean_color,
        _k("std_color"): config.std_color,
        _k("plot_info_box_manual"): bool(config.plot_info_box.manual),
        _k("plot_info_box_font_size"): int(config.plot_info_box.font_size),
        _k("plot_info_box_x"): float(st.session_state.get(_k("plot_info_box_x"), 0.02)),
        _k("plot_info_box_y"): float(st.session_state.get(_k("plot_info_box_y"), 0.98)),
        _k("plot_info_box_width"): float(st.session_state.get(_k("plot_info_box_width"), 0.46)),
        _k("plot_info_box_height"): float(st.session_state.get(_k("plot_info_box_height"), 0.28)),
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
        _k("autoscale_export_axes"): bool(export_config.autoscale_axes),
    }
