"""Small UI-facing helpers for labels and displayed equations."""

from __future__ import annotations

import re

import numpy as np

from .config import DEFAULT_X_LABEL, DEFAULT_Y_LABEL
from .i18n import translate

_TEX_UNICODE_MAP = {
    r"\Delta": "\u0394",
    r"\delta": "\u03b4",
    r"\sigma": "\u03c3",
    r"\Sigma": "\u03a3",
    r"\mu": "\u03bc",
    r"\lambda": "\u03bb",
    r"\Lambda": "\u039b",
    r"\pi": "\u03c0",
    r"\Pi": "\u03a0",
    r"\phi": "\u03c6",
    r"\Phi": "\u03a6",
    r"\omega": "\u03c9",
    r"\Omega": "\u03a9",
    r"\alpha": "\u03b1",
    r"\beta": "\u03b2",
    r"\gamma": "\u03b3",
    r"\tau": "\u03c4",
    r"\theta": "\u03b8",
    r"\rho": "\u03c1",
    r"\pm": "\u00b1",
    r"\cdot": "\u00b7",
    r"\times": "\u00d7",
}

_UNICODE_SYMBOL_MAP = {
    "\u0394": "\u0394",
    "\u03b4": "\u03b4",
    "\u03c3": "\u03c3",
    "\u03a3": "\u03a3",
    "\u03bc": "\u03bc",
    "\u03bb": "\u03bb",
    "\u039b": "\u039b",
    "\u03c0": "\u03c0",
    "\u03a0": "\u03a0",
    "\u03c6": "\u03c6",
    "\u03a6": "\u03a6",
    "\u03c9": "\u03c9",
    "\u03a9": "\u03a9",
    "\u03b1": "\u03b1",
    "\u03b2": "\u03b2",
    "\u03b3": "\u03b3",
    "\u03c4": "\u03c4",
    "\u03b8": "\u03b8",
    "\u03c1": "\u03c1",
    "\u00b1": "\u00b1",
    "\u00b7": "\u00b7",
    "\u00d7": "\u00d7",
}

_SUPERSCRIPT_CHARS = {
    "\u2070": "0",
    "\u00b9": "1",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u2074": "4",
    "\u2075": "5",
    "\u2076": "6",
    "\u2077": "7",
    "\u2078": "8",
    "\u2079": "9",
    "\u207a": "+",
    "\u207b": "-",
    "\u207c": "=",
    "\u207d": "(",
    "\u207e": ")",
    "\u2071": "i",
    "\u207f": "n",
}

_SUBSCRIPT_CHARS = {
    "\u2080": "0",
    "\u2081": "1",
    "\u2082": "2",
    "\u2083": "3",
    "\u2084": "4",
    "\u2085": "5",
    "\u2086": "6",
    "\u2087": "7",
    "\u2088": "8",
    "\u2089": "9",
    "\u208a": "+",
    "\u208b": "-",
    "\u208c": "=",
    "\u208d": "(",
    "\u208e": ")",
    "\u2090": "a",
    "\u2091": "e",
    "\u2095": "h",
    "\u1d62": "i",
    "\u2c7c": "j",
    "\u2096": "k",
    "\u2097": "l",
    "\u2098": "m",
    "\u2099": "n",
    "\u2092": "o",
    "\u209a": "p",
    "\u1d63": "r",
    "\u209b": "s",
    "\u209c": "t",
    "\u1d64": "u",
    "\u1d65": "v",
    "\u2093": "x",
}

_SUPERSCRIPT_PATTERN = re.compile("(" + "|".join(re.escape(char) for char in _SUPERSCRIPT_CHARS) + ")+")
_SUBSCRIPT_PATTERN = re.compile("(" + "|".join(re.escape(char) for char in _SUBSCRIPT_CHARS) + ")+")
_MATH_TRIGGER_PATTERN = re.compile(
    r"(\\[A-Za-z]+|[_^]|[\u0394\u03b4\u03c3\u03a3\u03bc\u03bb\u039b\u03c0\u03a0\u03c6\u03a6\u03c9\u03a9\u03b1\u03b2\u03b3\u03c4\u03b8\u03c1\u00b1\u00b7\u00d7]|[\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207a\u207b\u207c\u207d\u207e\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u208a\u208b\u208c\u208d\u208e\u2090\u2091\u2095\u1d62\u2c7c\u2096\u2097\u2098\u2099\u2092\u209a\u1d63\u209b\u209c\u1d64\u1d65\u2093])"
)


def safe_default_index(options: list[str], preferred: str | None, fallback: int = 0) -> int:
    """Return preferred option index if present, otherwise fallback."""
    if preferred and preferred in options:
        return options.index(preferred)
    return fallback


def _replace_tex_macros(text: str) -> str:
    """Replace common TeX macros with readable Unicode symbols."""
    out = str(text)
    for macro, symbol in _TEX_UNICODE_MAP.items():
        out = out.replace(macro, symbol)
    return out


def _replace_unicode_math_symbols(text: str) -> str:
    """Normalize direct Unicode math symbols while keeping them as symbols."""
    out = str(text)
    for unicode_char, symbol in _UNICODE_SYMBOL_MAP.items():
        out = out.replace(unicode_char, symbol)
    return out


def _replace_unicode_scripts(text: str) -> str:
    """Normalize Unicode superscripts/subscripts into ^{...}/_{...} form."""
    out = str(text)
    out = _SUPERSCRIPT_PATTERN.sub(
        lambda match: "^{" + "".join(_SUPERSCRIPT_CHARS[ch] for ch in match.group(0)) + "}",
        out,
    )
    out = _SUBSCRIPT_PATTERN.sub(
        lambda match: "_{" + "".join(_SUBSCRIPT_CHARS[ch] for ch in match.group(0)) + "}",
        out,
    )
    return out


def _convert_scripts_to_html(text: str) -> str:
    """Convert ^ / _ notation into HTML sup/sub tags for Plotly text rendering."""
    out = str(text)
    previous = None
    while out != previous:
        previous = out
        out = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", out)
        out = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", out)
        out = re.sub(r"\^([A-Za-z0-9+\-.]+)", r"<sup>\1</sup>", out)
        out = re.sub(r"_([A-Za-z0-9+\-.]+)", r"<sub>\1</sub>", out)
    return out


def prettify_plot_text(text: str) -> str:
    """Convert TeX-like math notation to Plotly-friendly Unicode and HTML."""
    out = str(text).strip()
    out = _replace_unicode_scripts(out)
    out = _replace_tex_macros(out)
    out = _replace_unicode_math_symbols(out)
    out = _convert_scripts_to_html(out)
    out = out.replace("{", "").replace("}", "")
    return out


def to_plot_math_text(text: str, use_latex: bool) -> str:
    """Convert math-style label input into Plotly-friendly rich text when enabled."""
    stripped = str(text).strip()
    if not stripped:
        return str(text)
    if not use_latex:
        return stripped
    if not _MATH_TRIGGER_PATTERN.search(stripped):
        return stripped
    return prettify_plot_text(stripped)


def squared_label_from_column(column_name: str) -> str:
    """Build readable squared label from a source column header."""
    label = str(column_name).strip()
    if not label:
        return "T\u00b2"
    match = re.match(r"^(.*?)\s*\[(.*?)\]\s*$", label)
    if match:
        quantity = match.group(1).strip() or "T"
        unit = match.group(2).strip()
        return f"({quantity})\u00b2 [{unit}\u00b2]"
    return f"({label})\u00b2"


def auto_axis_labels(
    x_column: str,
    y_column: str | None,
) -> tuple[str, str]:
    """Suggest axis labels based on selected explicit x/y columns."""
    x_label = str(x_column).strip() if str(x_column).strip() else DEFAULT_X_LABEL
    y_col = str(y_column).strip() if y_column else ""
    y_label = y_col if y_col else DEFAULT_Y_LABEL
    return x_label, y_label


def auto_line_labels(error_method: str, lang: str = "de") -> tuple[str, str, str]:
    """Suggest line labels based on selected error-line method."""
    fit_label = translate(lang, "fit.default_label")
    if error_method == "protocol":
        return (
            fit_label,
            translate(lang, "error.default_label.max.standard"),
            translate(lang, "error.default_label.min.standard"),
        )
    return (
        fit_label,
        translate(lang, "error.default_label.max.centroid"),
        translate(lang, "error.default_label.min.centroid"),
    )


def format_linear_equation(label: str, slope: float, intercept: float, decimals: int = 5) -> str:
    """Format y = a*x + b equation text."""
    sign = "+" if intercept >= 0 else "-"
    return f"{label}: y = {slope:.{int(decimals)}g}\u00b7x {sign} {abs(intercept):.{int(decimals)}g}"


def format_exponential_equation(
    label: str,
    slope_log: float,
    intercept_log: float,
    decimals: int = 5,
) -> str:
    """Format y = k*exp(a*x) equation text from log-space line params."""
    prefactor = float(np.exp(intercept_log))
    return f"{label}: y = {prefactor:.{int(decimals)}g}\u00b7e^({slope_log:.{int(decimals)}g}\u00b7x)"


def fit_line_help_text(fit_model: str, lang: str = "de") -> str:
    """Return tooltip text for the Ausgleichsgerade toggle."""
    if fit_model == "exp":
        return translate(lang, "tooltip.fit_exp")
    return translate(lang, "tooltip.fit_linear")


def error_line_help_text(fit_model: str, error_method: str, lang: str = "de") -> str:
    """Return tooltip text for the Fehlergeraden toggle."""
    model_note = translate(
        lang,
        "tooltip.error_model_note.exp" if fit_model == "exp" else "tooltip.error_model_note.linear",
    )

    if error_method == "protocol":
        return translate(lang, "tooltip.error_protocol", model_note=model_note)

    return translate(lang, "tooltip.error_centroid", model_note=model_note)


def _clean_label_for_symbol(label: str) -> str:
    """Extract the variable-like part from an axis label."""
    text = str(label).strip()
    if not text:
        return ""
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    text = re.sub(r"\[.*?\]", "", text).strip()
    text = text.rstrip(',:;')
    if text.startswith("(") and text.endswith(")") and len(text) > 2:
        text = text[1:-1].strip()
    return text


def auto_delta_symbol_from_label(label: str, fallback_axis: str) -> str:
    """Return auto-detected delta symbol from label, with axis fallback."""
    cleaned = _clean_label_for_symbol(label)
    if cleaned:
        return f"\u0394{cleaned}"
    return f"\u0394{fallback_axis.upper()}"


def auto_triangle_delta_symbols(x_label: str, y_label: str) -> tuple[str, str]:
    """Return auto-detected horizontal and vertical delta symbols."""
    return (
        auto_delta_symbol_from_label(x_label, "X"),
        auto_delta_symbol_from_label(y_label, "Y"),
    )
