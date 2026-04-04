"""Tests for statistics-mode helper calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.analysis_service import analyze_statistics_mode
from src.i18n import translate
from src.mode_models import PlotInfoBoxConfig, StatisticsModeConfig
from src.statistics import describe_distribution, normal_curve_points, normal_pdf


def test_describe_distribution_returns_sample_statistics() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])

    result = describe_distribution(values)

    assert result.count == 4
    assert result.mean == pytest.approx(2.5)
    assert result.variance == pytest.approx(5.0 / 3.0)
    assert result.std == pytest.approx(np.sqrt(5.0 / 3.0))
    assert result.median == pytest.approx(2.5)
    assert result.minimum == pytest.approx(1.0)
    assert result.maximum == pytest.approx(4.0)
    assert result.q1 == pytest.approx(1.75)
    assert result.q3 == pytest.approx(3.25)


def test_normal_pdf_matches_standard_normal_peak() -> None:
    pdf = normal_pdf(np.array([0.0]), mean=0.0, std=1.0)
    assert pdf[0] == pytest.approx(1.0 / np.sqrt(2.0 * np.pi))


def test_normal_curve_points_cover_sample_and_return_positive_density() -> None:
    values = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    x_grid, y_grid = normal_curve_points(values, mean=12.0, std=1.5, num_points=120)

    assert x_grid.size == 120
    assert y_grid.size == 120
    assert np.min(x_grid) <= np.min(values)
    assert np.max(x_grid) >= np.max(values)
    assert np.all(y_grid > 0)


def test_analyze_statistics_mode_formula_box_uses_math_symbols() -> None:
    values = np.array([9.8, 9.9, 10.0, 10.1, 10.2])
    df = pd.DataFrame({"measurement": values})
    config = StatisticsModeConfig(
        stats_column="measurement",
        bins=8,
        normalize_density=False,
        use_math_text=True,
        auto_axis_labels=True,
        x_label="measurement",
        y_label="count",
        base_font_size=14,
        axis_title_font_size=16,
        tick_font_size=12,
        annotation_font_size=12,
        plot_info_box=PlotInfoBoxConfig(manual=False, font_size=12),
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

    result = analyze_statistics_mode(df, config, translate=lambda key, **kwargs: translate("en", key, **kwargs))

    joined = "\n".join(result.plot_contract.plot_info_lines)
    assert "?" not in joined
    assert "μ" in joined
    assert "σ" in joined
    assert "π" in joined


def test_analyze_statistics_mode_preview_uses_padded_x_range() -> None:
    values = np.array([9.68, 9.74, 9.78, 9.82, 9.84, 9.88, 9.90, 9.92, 9.94, 9.96, 9.98, 10.00, 10.02, 10.06, 10.10, 10.18, 10.20])
    df = pd.DataFrame({"measurement": values})
    config = StatisticsModeConfig(
        stats_column="measurement",
        bins=16,
        normalize_density=False,
        use_math_text=True,
        auto_axis_labels=True,
        x_label="measurement",
        y_label="count",
        base_font_size=14,
        axis_title_font_size=16,
        tick_font_size=12,
        annotation_font_size=12,
        plot_info_box=PlotInfoBoxConfig(manual=False, font_size=12),
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

    result = analyze_statistics_mode(df, config, translate=lambda key, **kwargs: translate("en", key, **kwargs))
    x_range = list(result.plot_contract.preview_figure.layout.xaxis.range)

    assert x_range[0] < float(np.min(values))
    assert x_range[1] > float(np.max(values))
