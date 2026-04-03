"""Unit tests for core physics calculations."""

from __future__ import annotations

import numpy as np

from src.calculations import (
    centroid_error_lines,
    compute_sigma_y_from_t,
    compute_y_from_t,
    endpoint_extreme_error_lines,
    exponential_regression,
    free_intercept_error_lines,
    is_line_compatible,
    linear_regression,
    logarithmic_transform_with_uncertainty,
    protocol_endpoint_error_lines,
)


def test_compute_y_from_t_squared() -> None:
    t = np.array([1.0, 2.0, 3.5])
    expected = np.array([1.0, 4.0, 12.25])
    np.testing.assert_allclose(compute_y_from_t(t), expected)


def test_compute_sigma_y_from_t_propagation() -> None:
    t = np.array([1.0, 2.0, 3.0])
    sigma_t = np.array([0.1, 0.05, 0.2])
    expected = np.array([0.2, 0.2, 1.2])
    np.testing.assert_allclose(compute_sigma_y_from_t(t, sigma_t), expected)


def test_linear_regression_perfect_line() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    fit = linear_regression(x, y)

    assert fit.k_fit == 2.0
    assert fit.l_fit == 1.0


def test_exponential_regression_on_perfect_exponential() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 5.0 * np.exp(0.4 * x)
    fit = exponential_regression(x, y)

    assert abs(fit.k_fit - 0.4) < 1e-12
    assert abs(fit.l_fit - np.log(5.0)) < 1e-12


def test_logarithmic_transform_with_uncertainty() -> None:
    y = np.array([10.0, 20.0, 50.0])
    sigma_y = np.array([1.0, 2.0, 5.0])
    y_log, sigma_log = logarithmic_transform_with_uncertainty(y, sigma_y)

    np.testing.assert_allclose(y_log, np.log(y))
    np.testing.assert_allclose(sigma_log, sigma_y / y)


def test_centroid_error_lines_and_delta_k() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 5.0])
    sigma_y = np.array([0.2, 0.2, 0.2])

    result = centroid_error_lines(x, y, sigma_y)

    assert abs(result.k_min - 1.8) < 1e-12
    assert abs(result.k_max - 2.2) < 1e-12
    assert abs(result.delta_k - 0.2) < 1e-12


def test_error_line_compatibility_for_extreme_slopes() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 5.0])
    sigma_y = np.array([0.2, 0.2, 0.2])

    result = centroid_error_lines(x, y, sigma_y)

    assert is_line_compatible(x, y, sigma_y, result.k_min, result.l_min)
    assert is_line_compatible(x, y, sigma_y, result.k_max, result.l_max)
    assert not is_line_compatible(x, y, sigma_y, 2.3, 0.7)


def test_free_intercept_error_lines_for_tight_dataset() -> None:
    # Dataset similar to T^2(m) case where centroid-constrained bounds can be infeasible.
    x = np.array([50.0, 100.0, 150.0, 200.0, 250.0])
    y = np.array([0.8464, 1.4641, 2.1609, 2.7556, 3.4969])
    sigma_y = np.array([0.01288, 0.0242, 0.01764, 0.0332, 0.0374])

    with np.testing.assert_raises(ValueError):
        free_intercept_error_lines(x, y, sigma_y)

    subjective = endpoint_extreme_error_lines(x, y, sigma_y)
    assert subjective.k_max >= subjective.k_min


def test_protocol_endpoint_error_lines_returns_extremes() -> None:
    x = np.array([50.0, 100.0, 150.0, 200.0, 250.0])
    y = np.array([0.8464, 1.4641, 2.1609, 2.7556, 3.4969])
    sigma_y = np.array([0.01288, 0.0242, 0.01764, 0.0332, 0.0374])

    result = protocol_endpoint_error_lines(x, y, sigma_y)

    assert result.k_max >= result.k_min
