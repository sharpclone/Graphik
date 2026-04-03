"""Geometry helpers for slope triangles and deterministic line points."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    """Simple 2D point."""

    x: float
    y: float


def line_y(slope: float, intercept: float, x: float) -> float:
    """Evaluate y value on line y = slope*x + intercept."""
    return slope * x + intercept


def segment_endpoints_on_line(
    slope: float,
    intercept: float,
    x_min: float,
    x_max: float,
) -> tuple[Point, Point]:
    """Return two endpoint points on a line within [x_min, x_max]."""
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if x_max == x_min:
        raise ValueError("x_min and x_max must differ.")

    a = Point(x=x_min, y=line_y(slope, intercept, x_min))
    b = Point(x=x_max, y=line_y(slope, intercept, x_max))
    return a, b


def auto_triangle_points(
    slope: float,
    intercept: float,
    x_min: float,
    x_max: float,
    margin_fraction: float = 0.1,
) -> tuple[Point, Point]:
    """Select far-apart deterministic points for slope-triangle construction."""
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if x_max == x_min:
        raise ValueError("x_min and x_max must differ.")

    margin_fraction = max(0.0, min(0.45, margin_fraction))
    dx = x_max - x_min
    x_a = x_min + margin_fraction * dx
    x_b = x_max - margin_fraction * dx

    if x_a == x_b:
        x_a, x_b = x_min, x_max

    a = Point(x=x_a, y=line_y(slope, intercept, x_a))
    b = Point(x=x_b, y=line_y(slope, intercept, x_b))
    return a, b


def custom_points_from_x(
    slope: float,
    intercept: float,
    x_a: float,
    x_b: float,
) -> tuple[Point, Point]:
    """Construct user-defined points on a line by x-coordinates."""
    if x_a == x_b:
        raise ValueError("Point A and B x-values must differ.")

    a = Point(x=x_a, y=line_y(slope, intercept, x_a))
    b = Point(x=x_b, y=line_y(slope, intercept, x_b))
    return a, b


def right_triangle_corner(a: Point, b: Point) -> Point:
    """Return right-angle corner for triangle using horizontal + vertical legs."""
    return Point(x=b.x, y=a.y)


def triangle_deltas(a: Point, b: Point) -> tuple[float, float, float]:
    """Return delta_x, delta_y and slope from two points."""
    delta_x = b.x - a.x
    if delta_x == 0:
        raise ValueError("delta_x is zero; cannot compute slope.")
    delta_y = b.y - a.y
    slope = delta_y / delta_x
    return delta_x, delta_y, slope
