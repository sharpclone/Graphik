"""Descriptive statistics helpers for the statistics plotting mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DistributionStats:
    """Summary statistics for a one-dimensional numeric sample."""

    count: int
    mean: float
    std: float
    variance: float
    median: float
    minimum: float
    maximum: float
    q1: float
    q3: float


def _as_1d_float_array(values: Any) -> FloatArray:
    """Convert values to a finite 1D float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if array.size == 0:
        raise ValueError("values must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contain non-finite numbers.")
    return np.asarray(array, dtype=np.float64)


def describe_distribution(values: Any) -> DistributionStats:
    """Return descriptive statistics for a one-dimensional sample."""
    sample = _as_1d_float_array(values)
    count = int(sample.size)
    mean = float(np.mean(sample))
    variance = float(np.var(sample, ddof=1)) if count > 1 else 0.0
    std = float(np.sqrt(variance))
    median = float(np.median(sample))
    minimum = float(np.min(sample))
    maximum = float(np.max(sample))
    q1 = float(np.quantile(sample, 0.25))
    q3 = float(np.quantile(sample, 0.75))
    return DistributionStats(
        count=count,
        mean=mean,
        std=std,
        variance=variance,
        median=median,
        minimum=minimum,
        maximum=maximum,
        q1=q1,
        q3=q3,
    )


def normal_pdf(x: Any, mean: float, std: float) -> FloatArray:
    """Evaluate the normal probability density function."""
    x_arr = _as_1d_float_array(x)
    if std <= 0:
        raise ValueError("std must be positive for a normal density.")
    coeff = 1.0 / (float(std) * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * ((x_arr - float(mean)) / float(std)) ** 2
    return np.asarray(coeff * np.exp(exponent), dtype=np.float64)


def normal_curve_points(
    values: Any,
    mean: float,
    std: float,
    num_points: int = 400,
    sigma_padding: float = 3.0,
) -> tuple[FloatArray, FloatArray]:
    """Build a smooth x-grid and fitted normal density values."""
    sample = _as_1d_float_array(values)
    if std <= 0:
        raise ValueError("std must be positive for a normal density.")

    data_min = float(np.min(sample))
    data_max = float(np.max(sample))
    lower = min(data_min, float(mean) - float(sigma_padding) * float(std))
    upper = max(data_max, float(mean) + float(sigma_padding) * float(std))
    if upper <= lower:
        pad = max(1.0, abs(lower) * 0.05)
        lower -= pad
        upper += pad
    x_grid = np.linspace(lower, upper, int(max(50, num_points)), dtype=np.float64)
    return x_grid, normal_pdf(x_grid, mean=mean, std=std)
