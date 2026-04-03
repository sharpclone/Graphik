"""Unit tests for translation helpers."""

from __future__ import annotations

from src.i18n import translate


def test_translate_replaces_named_placeholders() -> None:
    assert translate("en", "data_source.loaded_file", filename="test.csv") == "Loaded test.csv"


def test_translate_keeps_math_braces_intact() -> None:
    text = translate("en", "plot_settings.use_latex_help")
    assert r"e^{-\mu x}" in text
