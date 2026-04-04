from __future__ import annotations

import numpy as np

from services.analysis_service import build_normal_analysis_result
from src.data_io import get_sample_dataframe
from src.i18n import translate
from src.mode_models import NormalModeConfig, PlotInfoBoxConfig


def _normal_config(*, extrapolate_lines: bool) -> NormalModeConfig:
    return NormalModeConfig(
        mapping_mode="simple",
        use_zero_error=False,
        x_column="m",
        y_column="y",
        sigma_y_column="sigma_y",
        use_math_text=True,
        auto_axis_labels=True,
        x_label="m [g]",
        y_label="T^2 [s^2]",
        y_axis_type="linear",
        y_log_decades=None,
        base_font_size=14,
        axis_title_font_size=14,
        tick_font_size=12,
        annotation_font_size=12,
        plot_info_box=PlotInfoBoxConfig(),
        show_grid=True,
        grid_mode="auto",
        x_major_divisions=10,
        y_major_divisions=10,
        minor_per_major=10,
        marker_size=7.0,
        error_bar_thickness=1.5,
        error_bar_cap_width=6.0,
        connect_points=False,
        x_tick_decimals=1,
        y_tick_decimals=2,
        x_range=(40.0, 260.0),
        y_range=(0.0, 3.8),
        fit_model="linear",
        show_fit_line=True,
        show_error_lines=False,
        extrapolate_lines=extrapolate_lines,
        error_line_mode="protocol",
        fit_label="Fit",
        fit_color="#ff0000",
        show_fit_slope_label=False,
        show_line_equations_on_plot=False,
        show_r2_on_plot=False,
        visible_error_lines_linear=(),
        visible_error_lines_exp=(),
        error_color="#00aa00",
        error_label_max="kmax",
        error_label_min="kmin",
        show_error_slope_label=False,
        show_fit_triangle=False,
        auto_fit_points=True,
        custom_fit_x_a=None,
        custom_fit_x_b=None,
        show_error_triangles=False,
        triangle_x_decimals=1,
        triangle_y_decimals=2,
        auto_line_labels_enabled=True,
    )


def test_normal_fit_line_stays_within_measured_x_bounds_when_extrapolation_disabled() -> None:
    result = build_normal_analysis_result(
        get_sample_dataframe(),
        _normal_config(extrapolate_lines=False),
        translate=lambda key, **kwargs: translate("en", key, **kwargs),
    )

    fit_trace = result.plot_contract.preview_figure.data[1]
    fit_x = np.asarray(fit_trace.x, dtype=float)

    assert fit_x.tolist() == [50.0, 250.0]


def test_normal_fit_line_uses_visible_x_bounds_when_extrapolation_enabled() -> None:
    result = build_normal_analysis_result(
        get_sample_dataframe(),
        _normal_config(extrapolate_lines=True),
        translate=lambda key, **kwargs: translate("en", key, **kwargs),
    )

    preview_fit_trace = result.plot_contract.preview_figure.data[1]
    export_fit_trace = result.plot_contract.build_export_base_figure(False).data[1]
    preview_x = np.asarray(preview_fit_trace.x, dtype=float)
    export_x = np.asarray(export_fit_trace.x, dtype=float)

    assert preview_x.tolist() == [40.0, 260.0]
    assert export_x.tolist() == [40.0, 260.0]
