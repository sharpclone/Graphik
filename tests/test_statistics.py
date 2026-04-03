"""Tests for statistics-mode helper calculations."""

from __future__ import annotations

import numpy as np
import pytest

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
