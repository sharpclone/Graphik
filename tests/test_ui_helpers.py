"""Unit tests for UI-facing math-label normalization."""

from __future__ import annotations

from src.ui_helpers import prettify_plot_text, to_plot_math_text


def test_to_plot_math_text_keeps_plain_text_without_math_markup() -> None:
    assert to_plot_math_text("Measured points", True) == "Measured points"


def test_to_plot_math_text_converts_delta_macro() -> None:
    assert to_plot_math_text(r"\Delta m [g]", True) == "Δ m [g]"


def test_to_plot_math_text_converts_subscript_and_superscript() -> None:
    assert to_plot_math_text("T_i [s]", True) == "T<sub>i</sub> [s]"
    assert to_plot_math_text("T^2 [s^2]", True) == "T<sup>2</sup> [s<sup>2</sup>]"


def test_to_plot_math_text_converts_unicode_scripts() -> None:
    assert to_plot_math_text("T² [s²]", True) == "T<sup>2</sup> [s<sup>2</sup>]"


def test_prettify_plot_text_handles_mixed_expression() -> None:
    assert prettify_plot_text(r"I_0 e^{-\mu x}") == "I<sub>0</sub> e<sup>-μ x</sup>"


def test_to_plot_math_text_returns_raw_input_when_disabled() -> None:
    assert to_plot_math_text(r"\sigma_T", False) == r"\sigma_T"
