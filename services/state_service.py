"""Session bootstrap, mode transitions, and legacy-state normalization helpers."""

from __future__ import annotations

from typing import Any, MutableMapping

import pandas as pd

from src.ui_state import (
    MODE_PREFIXES,
    NORMAL_MODE,
    STATISTICS_MODE,
    clear_mode_state,
    clear_session_snapshot,
    hydrate_mode_preferences,
    init_session_state,
    reset_corrupted_settings,
    reset_view_state,
    restore_session_snapshot,
    save_session_snapshot,
)


LEGACY_KEY_MIGRATIONS: dict[str, tuple[str, ...]] = {
    "use_latex_plot": ("normal.use_latex_plot", "stats.use_latex_plot"),
    "show_grid": ("normal.show_grid", "stats.show_grid"),
    "use_separate_fonts": ("normal.use_separate_fonts", "stats.use_separate_fonts"),
    "global_font_size": ("normal.global_font_size", "stats.global_font_size"),
    "base_font_size": ("normal.base_font_size", "stats.base_font_size"),
    "axis_title_font_size": ("normal.axis_title_font_size", "stats.axis_title_font_size"),
    "tick_font_size": ("normal.tick_font_size", "stats.tick_font_size"),
    "annotation_font_size": ("normal.annotation_font_size", "stats.annotation_font_size"),
    "plot_info_box_manual": ("normal.plot_info_box_manual", "stats.plot_info_box_manual"),
    "plot_info_box_font_size": ("normal.plot_info_box_font_size", "stats.plot_info_box_font_size"),
    "plot_info_box_x": ("normal.plot_info_box_x", "stats.plot_info_box_x"),
    "plot_info_box_y": ("normal.plot_info_box_y", "stats.plot_info_box_y"),
    "plot_info_box_width": ("normal.plot_info_box_width", "stats.plot_info_box_width"),
    "plot_info_box_height": ("normal.plot_info_box_height", "stats.plot_info_box_height"),
    "x_tick_decimals": ("normal.x_tick_decimals", "stats.x_tick_decimals"),
    "y_tick_decimals": ("normal.y_tick_decimals", "stats.y_tick_decimals"),
    "mapping_mode": ("normal.mapping_mode",),
    "use_zero_error": ("normal.use_zero_error",),
    "x_col_simple": ("normal.x_col_simple",),
    "y_col_simple": ("normal.y_col_simple",),
    "sigma_col_simple": ("normal.sigma_col_simple",),
    "x_col_adv": ("normal.x_col_adv",),
    "y_col_adv": ("normal.y_col_adv",),
    "sigma_y_col_adv": ("normal.sigma_y_col_adv",),
    "auto_axis_labels": ("normal.auto_axis_labels",),
    "x_label_input": ("normal.x_label_input",),
    "y_label_input": ("normal.y_label_input",),
    "y_scale_mode": ("normal.y_scale_mode",),
    "log_decade_mode": ("normal.log_decade_mode",),
    "grid_mode": ("normal.grid_mode",),
    "x_major_divisions": ("normal.x_major_divisions",),
    "y_major_divisions": ("normal.y_major_divisions",),
    "minor_per_major": ("normal.minor_per_major",),
    "marker_size": ("normal.marker_size",),
    "error_bar_thickness": ("normal.error_bar_thickness",),
    "error_bar_cap_width": ("normal.error_bar_cap_width",),
    "connect_points": ("normal.connect_points",),
    "custom_x_range": ("normal.custom_x_range",),
    "custom_y_range": ("normal.custom_y_range",),
    "x_min_user": ("normal.x_min_user",),
    "x_max_user": ("normal.x_max_user",),
    "y_min_user": ("normal.y_min_user",),
    "y_max_user": ("normal.y_max_user",),
    "fit_model": ("normal.fit_model",),
    "show_fit_line": ("normal.show_fit_line",),
    "show_error_lines": ("normal.show_error_lines",),
    "error_line_mode_widget": ("normal.error_line_mode_widget",),
    "auto_line_labels": ("normal.auto_line_labels",),
    "fit_label_input": ("normal.fit_label_input",),
    "fit_color": ("normal.fit_color",),
    "show_fit_slope_label": ("normal.show_fit_slope_label",),
    "show_line_equations_on_plot": ("normal.show_line_equations_on_plot",),
    "show_r2_on_plot": ("normal.show_r2_on_plot",),
    "visible_error_lines_exp": ("normal.visible_error_lines_exp",),
    "visible_error_lines_linear": ("normal.visible_error_lines_linear",),
    "error_color": ("normal.error_color",),
    "error_label_max_input": ("normal.error_label_max_input",),
    "error_label_min_input": ("normal.error_label_min_input",),
    "show_error_slope_label": ("normal.show_error_slope_label",),
    "show_fit_triangle": ("normal.show_fit_triangle",),
    "auto_fit_points": ("normal.auto_fit_points",),
    "fit_custom_ax": ("normal.fit_custom_ax",),
    "fit_custom_bx": ("normal.fit_custom_bx",),
    "show_error_triangles": ("normal.show_error_triangles",),
    "triangle_x_decimals": ("normal.triangle_x_decimals",),
    "triangle_y_decimals": ("normal.triangle_y_decimals",),
    "export_base": ("normal.export_base", "stats.export_base"),
    "png_paper": ("normal.png_paper", "stats.png_paper"),
    "png_orientation": ("normal.png_orientation", "stats.png_orientation"),
    "png_dpi": ("normal.png_dpi", "stats.png_dpi"),
    "png_scale": ("normal.png_scale", "stats.png_scale"),
    "png_word_like": ("normal.png_word_like", "stats.png_word_like"),
    "png_text_pt": ("normal.png_text_pt", "stats.png_text_pt"),
    "png_visual_scale": ("normal.png_visual_scale", "stats.png_visual_scale"),
    "autoscale_export_axes": ("normal.autoscale_export_axes",),
    "stats_column": ("stats.stats_column",),
    "stats_bins": ("stats.stats_bins",),
    "stats_normalize_density": ("stats.stats_normalize_density",),
    "stats_auto_axis_labels": ("stats.auto_axis_labels",),
    "stats_x_label_input": ("stats.x_label_input",),
    "stats_y_label_input": ("stats.y_label_input",),
    "stats_show_normal_fit": ("stats.show_normal_fit",),
    "stats_show_formula_box": ("stats.show_formula_box",),
    "stats_show_mean_line": ("stats.show_mean_line",),
    "stats_show_std_lines": ("stats.show_std_lines",),
    "stats_show_two_sigma": ("stats.show_two_sigma",),
    "stats_show_three_sigma": ("stats.show_three_sigma",),
    "stats_histogram_color": ("stats.histogram_color",),
    "stats_fit_color": ("stats.fit_color",),
    "stats_mean_color": ("stats.mean_color",),
    "stats_std_color": ("stats.std_color",),
}


def bootstrap_session_state(session_state: MutableMapping[str, Any], sample_dataframe: pd.DataFrame) -> None:
    """Restore persisted UI state and initialize required defaults."""
    restore_session_snapshot(session_state)
    init_session_state(session_state, sample_dataframe)


def normalize_legacy_selection(
    session_state: MutableMapping[str, Any],
    key: str,
    mapping: dict[str, str] | None = None,
) -> None:
    """Convert persisted legacy widget values to current internal option keys."""
    value = session_state.get(key)
    if isinstance(value, tuple) and value:
        session_state[key] = value[0]
        return
    if isinstance(value, str) and mapping and value in mapping:
        session_state[key] = mapping[value]


def migrate_legacy_widget_keys(session_state: MutableMapping[str, Any]) -> None:
    """Copy old non-namespaced widget keys into the current mode namespaces once."""
    for legacy_key, target_keys in LEGACY_KEY_MIGRATIONS.items():
        if legacy_key not in session_state:
            continue
        for target_key in target_keys:
            session_state.setdefault(target_key, session_state[legacy_key])


def apply_mode_switch_reset(session_state: MutableMapping[str, Any]) -> bool:
    """Clear mode-specific live state when the user switches between Normal and Statistics."""
    current_mode = str(session_state.get("app_mode", NORMAL_MODE))
    previous_mode = str(session_state.get("_last_app_mode", current_mode))
    if current_mode == previous_mode:
        return False

    if previous_mode in {NORMAL_MODE, STATISTICS_MODE}:
        clear_mode_state(session_state, previous_mode)
    for transient_key in [key for key in list(session_state.keys()) if key.startswith("_export_cache_")]:
        session_state.pop(transient_key, None)
    session_state.pop("_prefs", None)
    hydrate_mode_preferences(session_state, current_mode)
    session_state["_last_app_mode"] = current_mode
    return True


def reset_corrupted_settings_state(session_state: MutableMapping[str, Any], sample_dataframe: pd.DataFrame) -> None:
    """Expose a user-facing hard reset for persisted settings and runtime state."""
    reset_corrupted_settings(session_state, sample_dataframe)


__all__ = [
    "MODE_PREFIXES",
    "NORMAL_MODE",
    "STATISTICS_MODE",
    "apply_mode_switch_reset",
    "bootstrap_session_state",
    "clear_session_snapshot",
    "migrate_legacy_widget_keys",
    "normalize_legacy_selection",
    "reset_corrupted_settings_state",
    "reset_view_state",
    "save_session_snapshot",
]
