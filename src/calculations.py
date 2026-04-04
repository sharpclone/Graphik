"""Numerical calculations for fits, uncertainties, and plotting analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.optimize import linprog

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class LinearFitResult:
    """Result of a linear fit y = kx + l."""

    k_fit: float
    l_fit: float
    r_value: float
    p_value: float
    std_err: float


@dataclass(frozen=True)
class ErrorLineResult:
    """Result of centroid-constrained extreme slope lines."""

    x_bar: float
    y_bar: float
    k_min: float
    k_max: float
    l_min: float
    l_max: float
    delta_k: float


def _as_1d_float_array(values: Any, name: str) -> FloatArray:
    """Convert input values to a finite 1D float numpy array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return np.asarray(array, dtype=np.float64)


def _validate_same_length(*arrays: np.ndarray) -> None:
    """Ensure all provided arrays have equal length."""
    lengths = {arr.size for arr in arrays}
    if len(lengths) != 1:
        raise ValueError("All input arrays must have the same length.")


def linear_regression(x: Any, y: Any) -> LinearFitResult:
    """Perform linear regression y = kx + l."""
    x_arr = _as_1d_float_array(x, "x")
    y_arr = _as_1d_float_array(y, "y")
    _validate_same_length(x_arr, y_arr)
    if x_arr.size < 2:
        raise ValueError("At least two points are required for linear regression.")

    result = stats.linregress(x_arr, y_arr)
    return LinearFitResult(
        k_fit=float(result.slope),
        l_fit=float(result.intercept),
        r_value=float(result.rvalue),
        p_value=float(result.pvalue),
        std_err=float(result.stderr),
    )


def exponential_regression(x: Any, y: Any) -> LinearFitResult:
    """
    Fit exponential model y = k * exp(a*x) by linearizing ln(y) = a*x + ln(k).

    Returns slope/intercept in log-space:
    - k_fit -> a
    - l_fit -> ln(k)
    """
    x_arr = _as_1d_float_array(x, "x")
    y_arr = _as_1d_float_array(y, "y")
    _validate_same_length(x_arr, y_arr)
    if np.any(y_arr <= 0):
        raise ValueError("Exponential regression requires y > 0 for all points.")
    return linear_regression(x_arr, np.log(y_arr))


def logarithmic_transform_with_uncertainty(
    y: Any,
    sigma_y: Any,
) -> tuple[FloatArray, FloatArray]:
    """
    Convert y and sigma_y to log-space for semilog line/error analysis.

    Uses first-order propagation:
      z = ln(y), sigma_z = sigma_y / y
    """
    y_arr = _as_1d_float_array(y, "y")
    sigma_arr = _as_1d_float_array(sigma_y, "sigma_y")
    _validate_same_length(y_arr, sigma_arr)
    if np.any(y_arr <= 0):
        raise ValueError("Log transform requires y > 0 for all points.")
    if np.any(sigma_arr < 0):
        raise ValueError("sigma_y must be non-negative.")
    return np.log(y_arr), sigma_arr / y_arr


def line_value(x: Any, slope: float, intercept: float) -> np.ndarray:
    """Evaluate line y = slope * x + intercept."""
    x_arr = _as_1d_float_array(x, "x")
    return slope * x_arr + intercept


def is_line_compatible(
    x: Any,
    y: Any,
    sigma_y: Any,
    slope: float,
    intercept: float,
    tol: float = 1e-12,
) -> bool:
    """Check if line lies within all vertical error bars."""
    x_arr = _as_1d_float_array(x, "x")
    y_arr = _as_1d_float_array(y, "y")
    sigma_arr = _as_1d_float_array(sigma_y, "sigma_y")
    _validate_same_length(x_arr, y_arr, sigma_arr)
    if np.any(sigma_arr < 0):
        raise ValueError("sigma_y must be non-negative.")

    y_line = line_value(x_arr, slope=slope, intercept=intercept)
    lower = y_arr - sigma_arr - tol
    upper = y_arr + sigma_arr + tol
    return bool(np.all((y_line >= lower) & (y_line <= upper)))


def centroid_error_lines(x: Any, y: Any, sigma_y: Any, tol: float = 1e-12) -> ErrorLineResult:
    """
    Compute centroid-constrained extreme slopes compatible with vertical error bars.

    The lines are constrained to pass through centroid (x̄, ȳ).
    """
    x_arr = _as_1d_float_array(x, "x")
    y_arr = _as_1d_float_array(y, "y")
    sigma_arr = _as_1d_float_array(sigma_y, "sigma_y")
    _validate_same_length(x_arr, y_arr, sigma_arr)

    if x_arr.size < 2:
        raise ValueError("At least two points are required for error lines.")
    if np.any(sigma_arr < 0):
        raise ValueError("sigma_y must be non-negative.")

    x_bar = float(np.mean(x_arr))
    y_bar = float(np.mean(y_arr))

    lower_bound = -np.inf
    upper_bound = np.inf

    for x_i, y_i, s_i in zip(x_arr, y_arr, sigma_arr, strict=True):
        dx = x_i - x_bar
        y_low = y_i - s_i
        y_high = y_i + s_i

        if abs(dx) <= tol:
            if not (y_low - tol <= y_bar <= y_high + tol):
                raise ValueError(
                    "No centroid-compatible line exists: centroid violates error interval at x = x̄."
                )
            continue

        k1 = (y_low - y_bar) / dx
        k2 = (y_high - y_bar) / dx

        point_lower = min(k1, k2)
        point_upper = max(k1, k2)

        lower_bound = max(lower_bound, point_lower)
        upper_bound = min(upper_bound, point_upper)

    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        raise ValueError("Cannot determine finite slope bounds for error lines.")

    if lower_bound > upper_bound + tol:
        raise ValueError("No compatible centroid-constrained error lines found.")

    k_min = float(lower_bound)
    k_max = float(upper_bound)
    l_min = float(y_bar - k_min * x_bar)
    l_max = float(y_bar - k_max * x_bar)

    return ErrorLineResult(
        x_bar=x_bar,
        y_bar=y_bar,
        k_min=k_min,
        k_max=k_max,
        l_min=l_min,
        l_max=l_max,
        delta_k=float((k_max - k_min) / 2.0),
    )


def free_intercept_error_lines(x: Any, y: Any, sigma_y: Any) -> ErrorLineResult:
    """
    Compute extreme compatible slopes without forcing lines through the centroid.

    Solve two linear programs over variables [k, l]:
      y_i - sigma_i <= k*x_i + l <= y_i + sigma_i
    then maximize and minimize k.
    """
    x_arr = _as_1d_float_array(x, "x")
    y_arr = _as_1d_float_array(y, "y")
    sigma_arr = _as_1d_float_array(sigma_y, "sigma_y")
    _validate_same_length(x_arr, y_arr, sigma_arr)

    if x_arr.size < 2:
        raise ValueError("At least two points are required for error lines.")
    if np.any(sigma_arr < 0):
        raise ValueError("sigma_y must be non-negative.")

    # Upper:  k*x_i + l <= y_i + sigma_i
    a_upper = np.column_stack((x_arr, np.ones_like(x_arr)))
    b_upper = y_arr + sigma_arr

    # Lower: -(k*x_i + l) <= -(y_i - sigma_i)
    a_lower = np.column_stack((-x_arr, -np.ones_like(x_arr)))
    b_lower = -(y_arr - sigma_arr)

    a_ub = np.vstack((a_upper, a_lower))
    b_ub = np.concatenate((b_upper, b_lower))

    bounds = [(None, None), (None, None)]  # k, l unbounded

    # Minimize k
    res_min = linprog(c=np.array([1.0, 0.0]), A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    # Maximize k <=> minimize -k
    res_max = linprog(c=np.array([-1.0, 0.0]), A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not res_min.success or not res_max.success:
        raise ValueError("No compatible error lines found, even with free intercept.")

    k_min = float(res_min.x[0])
    l_min = float(res_min.x[1])
    k_max = float(res_max.x[0])
    l_max = float(res_max.x[1])

    x_bar = float(np.mean(x_arr))
    y_bar = float(np.mean(y_arr))

    return ErrorLineResult(
        x_bar=x_bar,
        y_bar=y_bar,
        k_min=k_min,
        k_max=k_max,
        l_min=l_min,
        l_max=l_max,
        delta_k=float((k_max - k_min) / 2.0),
    )


def endpoint_extreme_error_lines(x: Any, y: Any, sigma_y: Any) -> ErrorLineResult:
    """
    Build subjective extreme lines from endpoint error intervals.

    This method does not guarantee compatibility with all points. It is used as
    a last-resort visual fallback when strict compatible lines do not exist.
    """
    x_arr = _as_1d_float_array(x, "x")
    y_arr = _as_1d_float_array(y, "y")
    sigma_arr = _as_1d_float_array(sigma_y, "sigma_y")
    _validate_same_length(x_arr, y_arr, sigma_arr)

    if x_arr.size < 2:
        raise ValueError("At least two points are required for error lines.")
    if np.any(sigma_arr < 0):
        raise ValueError("sigma_y must be non-negative.")

    order = np.argsort(x_arr)
    xs = x_arr[order]
    ys = y_arr[order]
    ss = sigma_arr[order]

    x_left, x_right = float(xs[0]), float(xs[-1])
    if x_right == x_left:
        raise ValueError("Need at least two distinct x-values for endpoint error lines.")

    y_left, s_left = float(ys[0]), float(ss[0])
    y_right, s_right = float(ys[-1]), float(ss[-1])

    # Max slope: left at lower edge, right at upper edge.
    k_max = ((y_right + s_right) - (y_left - s_left)) / (x_right - x_left)
    l_max = (y_left - s_left) - k_max * x_left

    # Min slope: left at upper edge, right at lower edge.
    k_min = ((y_right - s_right) - (y_left + s_left)) / (x_right - x_left)
    l_min = (y_left + s_left) - k_min * x_left

    x_bar = float(np.mean(x_arr))
    y_bar = float(np.mean(y_arr))

    return ErrorLineResult(
        x_bar=x_bar,
        y_bar=y_bar,
        k_min=float(k_min),
        k_max=float(k_max),
        l_min=float(l_min),
        l_max=float(l_max),
        delta_k=float((k_max - k_min) / 2.0),
    )


def protocol_endpoint_error_lines(x: Any, y: Any, sigma_y: Any) -> ErrorLineResult:
    """
    Protocol endpoint method for subjective error lines.

    Uses the first and last x-points with endpoint uncertainty extremes:
    - one extreme line through (x_left, y_left + sigma_left) and (x_right, y_right - sigma_right)
    - the other through (x_left, y_left - sigma_left) and (x_right, y_right + sigma_right)

    For positive trends, this yields the minimal and maximal slope lines commonly
    used in lab protocols.
    """
    return endpoint_extreme_error_lines(x, y, sigma_y)


def format_final_slope(k_fit: float, delta_k: float, precision: int = 6) -> str:
    """Return formatted slope with uncertainty."""
    return f"a = {k_fit:.{precision}g} ± {delta_k:.{precision}g}"
