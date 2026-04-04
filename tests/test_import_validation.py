"""Tests for import wizard helpers and UX validation services."""

from __future__ import annotations

import pandas as pd

from services.import_service import apply_import_selection, build_import_wizard_result
from services.validation_service import (
    collect_import_wizard_problems,
    collect_normal_mode_problems,
    collect_statistics_mode_problems,
    has_blocking_problems,
)
from src.data_io import apply_header_strategy
from src.i18n import translate
from src.mode_models import NormalModeConfig, PlotInfoBoxConfig, StatisticsModeConfig


def _t(key: str, **kwargs: object) -> str:
    return translate("en", key, **kwargs)


def _normal_config(**overrides: object) -> NormalModeConfig:
    base = dict(
        mapping_mode="simple",
        use_zero_error=False,
        x_column="x",
        y_column="y",
        sigma_y_column="sigma_y",
        use_math_text=True,
        auto_axis_labels=True,
        x_label="x",
        y_label="y",
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
        x_range=None,
        y_range=None,
        fit_model="linear",
        show_fit_line=True,
        show_error_lines=True,
        extrapolate_lines=True,
        error_line_mode="protocol",
        fit_label="Fit",
        fit_color="#ff0000",
        show_fit_slope_label=True,
        show_line_equations_on_plot=False,
        show_r2_on_plot=False,
        visible_error_lines_linear=("k_max",),
        visible_error_lines_exp=(),
        error_color="#00aa00",
        error_label_max="kmax",
        error_label_min="kmin",
        show_error_slope_label=True,
        show_fit_triangle=True,
        auto_fit_points=True,
        custom_fit_x_a=None,
        custom_fit_x_b=None,
        show_error_triangles=True,
        triangle_x_decimals=1,
        triangle_y_decimals=2,
        auto_line_labels_enabled=True,
    )
    base.update(overrides)
    return NormalModeConfig(**base)


def _stats_config(**overrides: object) -> StatisticsModeConfig:
    base = dict(
        stats_column="measurement",
        bins=16,
        normalize_density=False,
        use_math_text=True,
        auto_axis_labels=True,
        x_label="measurement",
        y_label="Count",
        base_font_size=14,
        axis_title_font_size=14,
        tick_font_size=12,
        annotation_font_size=12,
        plot_info_box=PlotInfoBoxConfig(),
        show_grid=True,
        x_tick_decimals=2,
        y_tick_decimals=2,
        show_normal_fit=True,
        show_formula_box=True,
        show_mean_line=True,
        show_std_lines=True,
        show_two_sigma=False,
        show_three_sigma=False,
        histogram_color="#7aa6ff",
        fit_color="#d62728",
        mean_color="#222222",
        std_color="#2ca02c",
    )
    base.update(overrides)
    return StatisticsModeConfig(**base)


def test_apply_header_strategy_promotes_selected_row() -> None:
    source = pd.DataFrame(
        [
            ["metadata", None, None],
            ["Mass [g]", "T^2 [s^2]", "sigma_y"],
            [50, 0.8464, 0.0129],
            [100, 1.4641, 0.0242],
        ],
        columns=["Unnamed: 0", "Unnamed: 1", "Unnamed: 2"],
    )

    out = apply_header_strategy(source, header_mode="row", header_row_index=1)

    assert list(out.columns) == ["Mass [g]", "T^2 [s^2]", "sigma_y"]
    assert out.iloc[0].tolist() == [50, 0.8464, 0.0129]


def test_build_import_wizard_result_suggests_columns() -> None:
    raw_df = pd.DataFrame({"m": [50, 100], "y": [1.0, 2.0], "sigma_y": [0.1, 0.1]})

    result = build_import_wizard_result(
        raw_df,
        filename="demo.csv",
        signature="demo-1",
        header_mode="current",
        header_row_index=None,
    )

    assert result.suggested_x_column == "m"
    assert result.suggested_y_column == "y"
    assert result.suggested_sigma_y_column == "sigma_y"


def test_apply_import_selection_seeds_normal_namespace() -> None:
    session_state: dict[str, object] = {}
    imported_df = pd.DataFrame({"m": [50], "y": [1.0], "sigma_y": [0.1]})

    apply_import_selection(
        session_state,
        imported_df=imported_df,
        signature="sig-1",
        mode="normal",
        x_column="m",
        y_column="y",
        sigma_y_column="sigma_y",
        stats_column=None,
        use_zero_error=False,
    )

    assert session_state["uploaded_signature"] == "sig-1"
    assert session_state["normal.x_col_simple"] == "m"
    assert session_state["normal.y_col_adv"] == "y"
    assert session_state["normal.sigma_y_col_adv"] == "sigma_y"


def test_collect_normal_mode_problems_reports_exp_constraints() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1.0, 0.0, 2.0], "sigma_y": [0.1, 0.1, 0.1]})
    config = _normal_config(fit_model="exp")

    problems = collect_normal_mode_problems(df, config, translate=_t)

    assert any(problem.code == "exp_unavailable" for problem in problems)
    assert has_blocking_problems(problems)


def test_collect_normal_mode_problems_reports_log_axis_and_clip() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0.5, 1.0, 2.0], "sigma_y": [0.7, 0.1, 0.1]})
    config = _normal_config(y_axis_type="log")

    problems = collect_normal_mode_problems(df, config, translate=_t)

    codes = {problem.code for problem in problems}
    assert "log_error_clip" in codes


def test_collect_statistics_mode_problems_reports_constant_data() -> None:
    df = pd.DataFrame({"measurement": [5.0, 5.0, 5.0, 5.0]})
    config = _stats_config()

    problems = collect_statistics_mode_problems(df, config, translate=_t)

    assert any(problem.code == "constant_data" for problem in problems)


def test_collect_import_wizard_problems_requires_sigma_when_enabled() -> None:
    df = pd.DataFrame({"m": [50, 100], "y": [1.0, 2.0], "sigma_y": [0.1, 0.1]})

    problems = collect_import_wizard_problems(
        df,
        mode="normal",
        x_column="m",
        y_column="y",
        sigma_y_column=None,
        use_zero_error=False,
        stats_column=None,
        translate=_t,
    )

    assert any(problem.code == "wizard_missing_sigma" for problem in problems)
    assert has_blocking_problems(problems)
